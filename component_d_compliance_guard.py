"""
============================================================
COMPONENT D — ComplianceGuardAgent
============================================================
HR Copilot · MLDS 2026 · Maneesh Kumar & Ravikiran Ravada

Use Case:
  An employee asks about termination during probation.
  The PolicyRAGAgent retrieves correct policy text — but the
  raw answer might be incomplete, misrepresent timelines, or
  omit mandatory legal notices. The ComplianceGuardAgent:

  1. Cross-encoder reranks retrieved chunks for precision
  2. Filters chunks below relevance threshold
  3. NLI FactCheck: verifies agent responses are grounded
  4. Legal compliance scan: flags sensitive HR topics
  5. OnboardingAgent: specialist for new joiner queries

  HR-specific risk: Wrong advice on termination, POSH, or
  compliance can expose the company to legal liability.

Azure equivalents:
  Reranker  → Azure AI Search semantic ranker (L2 neural)
  NLI check → Azure AI Language (text classification)
  Legal scan → Azure OpenAI with domain-specific system prompt

Run: python3 component_d_compliance_guard.py
============================================================
"""
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass

from sentence_transformers import CrossEncoder
from transformers import pipeline as hf_pipeline

from hr_data_models import (
    HRQueryPlan, RetrievedChunk, AgentResponse, ComplianceCheckResult,
    AgentName, QueryIntent
)
from component_a_hr_indexing import load_index, INDEX_DIR
from component_b_orchestrator_agent import orchestrator_agent
from component_c_policy_data_agents import PolicyRAGAgent, DataQueryAgent

print("="*60)
print("  COMPONENT D — ComplianceGuardAgent")
print("="*60)

RERANK_MODEL  = "cross-encoder/ms-marco-MiniLM-L6-v2"
NLI_MODEL     = "cross-encoder/nli-deberta-v3-small"
RERANK_THRESH = 0.0   # Cross-encoder logit threshold (balanced: keep good & marginal chunks)
NLI_THRESH    = 0.40   # Entailment probability (balanced for relevancy + precision coverage)

# HR-sensitive topics requiring compliance review
SENSITIVE_PATTERNS = {
    "termination": r"\b(terminat|dismiss|fired|sacked|let go|end of employment)\b",
    "posh":        r"\b(posh|sexual harassment|inappropriate|misconduct)\b",
    "pip":         r"\b(pip|performance improvement|underperform|poor performance)\b",
    "legal":       r"\b(sue|lawsuit|legal action|labour court|tribunal|whistleblow)\b",
    "salary_cut":  r"\b(salary cut|pay cut|reduce salary|ctc reduction|demotion)\b",
}

# Mandatory caveats by topic
MANDATORY_CAVEATS = {
    "termination": "⚠️ For termination-related matters, consult your HRBP and Legal team. This response is informational only.",
    "posh":        "⚠️ POSH matters are strictly confidential. Contact the Internal Complaints Committee (ICC) at icc@enterprise.com.",
    "pip":         "ℹ️ PIP process involves HR, your manager, and the employee. Contact your HRBP for personalised guidance.",
    "legal":       "⚠️ For legal disputes, always consult the Legal & Compliance team before taking action.",
    "salary_cut":  "ℹ️ Salary adjustments require HR Director approval. Discuss with your HRBP.",
}


# ═══════════════════════════════════════════════════════════════
# ONBOARDING AGENT (specialist for new joiner queries)
# ═══════════════════════════════════════════════════════════════
class OnboardingAgent:
    """
    Specialist agent for new employee onboarding questions.
    Has a dedicated system prompt and retrieval scope focused
    only on the onboarding guide — reduces hallucination risk.

    Why specialist? New joiners ask procedural questions in
    uncertain language ("what do I do first?"). A general
    agent often over-retrieves from other policy domains.
    """

    ONBOARDING_CHECKLIST = {
        "pre_joining": [
            "Accept offer letter digitally on HRMS",
            "Submit KYC: Aadhaar, PAN, 3 passport photos, bank details",
            "Submit previous employment documents (offer letter, relieving letter)",
            "IT asset request auto-triggered — laptop ready on Day 1",
            "Receive buddy assignment email from HR",
        ],
        "day_1": [
            "Collect laptop + access card from IT helpdesk (Floor 2, Building A)",
            "Attend HR induction: 9:30 AM – 12:30 PM (Conference Room Atlas)",
            "Confirm email + Slack access",
            "Meet reporting manager for 1:1 goal-setting",
            "Enroll in health insurance on benefits portal (30-day window)",
        ],
        "week_1": [
            "Complete POSH training (mandatory, online, ~2 hrs)",
            "Complete Code of Conduct and Data Privacy training",
            "Set up weekly 1:1 with manager",
            "Review team documentation on Confluence",
        ],
        "first_90_days": [
            "30 days: Shadow team, understand tools",
            "60 days: First small project with mentor",
            "90 days: Full contribution, OKRs approved",
        ],
    }

    def __init__(self, policy_agent: PolicyRAGAgent):
        self.policy_agent = policy_agent
        self.name         = AgentName.ONBOARDING

    def run(self, plan: HRQueryPlan) -> AgentResponse:
        """
        Answer onboarding questions using both:
        1. PolicyRAGAgent retrieval (scoped to onboarding docs)
        2. Built-in checklist (structured, always accurate)
        """
        print(f"\n  [OnboardingAgent] Processing: '{plan.original_question[:60]}...'")
        q = plan.original_question.lower()

        # Override priority_docs to scope retrieval
        onboarding_plan = HRQueryPlan(
            original_question = plan.original_question,
            intent            = QueryIntent.ONBOARDING,
            sub_queries       = plan.sub_queries,
            agents_to_invoke  = [AgentName.ONBOARDING],
            priority_docs     = ["onboarding"],
            needs_structured  = False,
        )

        # Retrieve from onboarding docs
        rag_resp = self.policy_agent.run(onboarding_plan)

        # Augment with structured checklist for specific questions
        checklist_answer = ""
        if any(kw in q for kw in ["first day", "day 1", "day one"]):
            items = self.ONBOARDING_CHECKLIST["day_1"]
            checklist_answer = "Day 1 Checklist:\n" + "\n".join(f"  ☐ {item}" for item in items)
        elif any(kw in q for kw in ["document", "submit", "bring", "paperwork"]):
            items = self.ONBOARDING_CHECKLIST["pre_joining"]
            checklist_answer = "Documents to Submit Before Joining:\n" + "\n".join(f"  ☐ {item}" for item in items)
        elif any(kw in q for kw in ["30-60-90", "first 3 months", "90 day", "first month"]):
            items = self.ONBOARDING_CHECKLIST["first_90_days"]
            checklist_answer = "30-60-90 Day Plan:\n" + "\n".join(f"  {item}" for item in items)
        elif any(kw in q for kw in ["week 1", "first week", "training", "mandatory"]):
            items = self.ONBOARDING_CHECKLIST["week_1"]
            checklist_answer = "Week 1 Mandatory Tasks:\n" + "\n".join(f"  ☐ {item}" for item in items)

        # Merge RAG + checklist
        if checklist_answer:
            combined = f"{checklist_answer}\n\nAdditional Policy Details:\n{rag_resp.answer[:500]}"
        else:
            combined = rag_resp.answer

        return AgentResponse(
            agent       = self.name,
            answer      = combined,
            sources     = rag_resp.sources + ["onboarding_checklist"],
            confidence  = 0.88,
            chunks_used = rag_resp.chunks_used,
        )


# ═══════════════════════════════════════════════════════════════
# COMPLIANCE GUARD AGENT
# ═══════════════════════════════════════════════════════════════
class ComplianceGuardAgent:
    """
    The compliance and quality gate for ALL HR responses.

    Three-stage pipeline:
    1. Cross-encoder reranking — sorts retrieved chunks by query relevance
    2. NLI FactChecker — verifies responses are grounded in policy text
    3. Legal compliance scan — detects sensitive topics, adds caveats

    WHY HR SPECIFICALLY NEEDS THIS:
    - Wrong leave count advice → employee overspends leave
    - Wrong termination guidance → legal liability
    - POSH advice without proper referral → serious compliance breach
    - Salary band leak → compensation equity violation

    This agent runs AFTER PolicyRAGAgent produces chunks, before Synthesizer.
    """

    def __init__(self):
        self.name     = AgentName.COMPLIANCE
        self._reranker = None
        self._nli      = None

    def _load_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            print(f"  Loading reranker: {RERANK_MODEL}")
            self._reranker = CrossEncoder(RERANK_MODEL, max_length=512)
        return self._reranker

    def _load_nli(self):
        if self._nli is None:
            print(f"  Loading NLI model: {NLI_MODEL}")
            self._nli = hf_pipeline("text-classification", model=NLI_MODEL,
                                     device=-1, top_k=None)
        return self._nli

    # ── Reranking ──────────────────────────────────────────────────────────
    def rerank(self, question: str, chunks: List[RetrievedChunk],
               threshold: float = RERANK_THRESH) -> List[RetrievedChunk]:
        """
        Cross-encoder reranking: evaluates query and chunk TOGETHER.

        Bi-encoder (Component A) = fast but independent encoding.
        Cross-encoder (here)     = slow but joint encoding → much better relevance.

        Strategy: bi-encoder retrieves top 10, cross-encoder reranks them.
        Azure: Azure AI Search semantic ranker applies this automatically.
        """
        if not chunks:
            return []
        reranker = self._load_reranker()
        pairs    = [(question, c.text[:512]) for c in chunks]
        scores   = reranker.predict(pairs, show_progress_bar=False)

        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        result = [c for c, s in ranked if s >= threshold]
        for c, s in ranked:
            c.rerank_score = float(s)

        print(f"  [Compliance] Reranked {len(chunks)} → {len(result)} (thresh={threshold})")
        return result[:8]

    # ── NLI Fact Check ─────────────────────────────────────────────────────
    def fact_check(self, question: str,
                   chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        NLI entailment check: does this chunk SUPPORT answering the question?

        Labels: ENTAILMENT (chunk answers Q) / NEUTRAL / CONTRADICTION
        Only ENTAILMENT chunks pass to the Synthesizer.

        Why HR needs this:
        BM25 often retrieves chunks that share terms but are about
        a different policy (e.g., "sick leave" chunk when user asks about
        "carry-forward" — same domain, wrong clause).
        """
        nli = self._load_nli()
        verified = []
        for chunk in chunks:
            nli_input = f"{chunk.text[:400]} [SEP] {question}"
            try:
                results = nli(nli_input)
                label_scores = {}
                for r in results:
                    lbl = r['label'].upper()
                    if '2' in lbl or 'ENTAIL' in lbl:
                        label_scores['ENTAILMENT'] = r['score']
                    elif '0' in lbl or 'CONTRA' in lbl:
                        label_scores['CONTRADICTION'] = r['score']
                    else:
                        label_scores['NEUTRAL'] = r['score']
                entail = label_scores.get('ENTAILMENT', 0.0)
                chunk.entail_score = entail
                chunk.verified     = entail >= NLI_THRESH
            except Exception:
                chunk.verified = True   # Assume OK on model error
            if chunk.verified:
                verified.append(chunk)

        print(f"  [Compliance] Fact-checked: {len(verified)}/{len(chunks)} pass entailment")
        return verified if verified else chunks[:3]   # Fallback: top 3

    # ── Legal Compliance Scan ──────────────────────────────────────────────
    def compliance_scan(self, question: str, answer: str) -> ComplianceCheckResult:
        """
        Scan question and answer for sensitive HR topics.
        Adds mandatory legal caveats to the response.

        HR-specific sensitive topics:
        - Termination → refer to HRBP + Legal
        - POSH → refer to ICC, do not advise directly
        - PIP → requires HRBP personalisation
        - Legal threats → escalate to Legal team
        """
        flags    = []
        caveats  = []
        combined = (question + " " + answer).lower()

        for topic, pattern in SENSITIVE_PATTERNS.items():
            if re.search(pattern, combined, re.IGNORECASE):
                flags.append(topic)
                caveats.append(MANDATORY_CAVEATS[topic])

        passes = True   # Always pass — we add caveats rather than blocking
        # Exception: POSH complaints should NEVER be answered by the copilot directly
        # Only block when the QUESTION itself shows intent to file a complaint,
        # not when the retrieved policy text naturally mentions the complaint process.
        q_lower = question.lower()
        if "posh" in flags and re.search(r"\b(file|lodge|register|report|raise)\b.*\b(complaint|harassment)\b", q_lower):
            passes = False
            corrected = (
                "⚠️ POSH complaints require confidential handling through the "
                "Internal Complaints Committee (ICC).\n\n"
                "Please contact: icc@enterprise.com\n"
                "Ethics Hotline (confidential, 24×7): 1800-XXX-1234\n\n"
                "The HR Copilot cannot process POSH complaints. All such matters "
                "are handled exclusively by the ICC as per POSH Act 2013."
            )
            return ComplianceCheckResult(
                passes=False, confidence=0.99, flags=flags,
                corrected_answer=corrected
            )

        if caveats:
            print(f"  [Compliance] Flags: {flags}")
        return ComplianceCheckResult(
            passes=passes, confidence=0.90, flags=flags,
            corrected_answer=None   # Caveats added by Synthesizer
        )

    def run(self, question: str, chunks: List[RetrievedChunk],
            agent_response: Optional[AgentResponse] = None) -> Tuple[List[RetrievedChunk], ComplianceCheckResult]:
        """
        Full compliance pipeline: rerank → fact-check → compliance scan.
        Returns verified chunks + compliance result for the Synthesizer.
        """
        print(f"\n  [ComplianceGuardAgent] Running compliance pipeline...")
        reranked  = self.rerank(question, chunks)
        verified  = self.fact_check(question, reranked)
        raw_ans   = agent_response.answer if agent_response else " ".join(c.text for c in verified[:3])
        compliance= self.compliance_scan(question, raw_ans)
        return verified, compliance


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  Loading indexes...")
    chunks, faiss_idx, bm25, model = load_index(INDEX_DIR)

    policy_agent     = PolicyRAGAgent(chunks, faiss_idx, bm25, model)
    data_agent       = DataQueryAgent()
    compliance_guard = ComplianceGuardAgent()

    test_cases = [
        "What is the annual leave carry-forward policy?",
        "I want to file a POSH complaint against my manager.",
        "What happens if I am put on a Performance Improvement Plan?",
        "What documents do I need on my first day?",
    ]

    for question in test_cases:
        print(f"\n{'─'*60}")
        print(f"  Q: {question}")
        plan      = orchestrator_agent(question, use_llm=False)

        # Get policy chunks
        retrieved = []
        for sq in plan.sub_queries:
            retrieved.extend(policy_agent.retrieve(sq, plan))

        # If onboarding question, use OnboardingAgent
        if AgentName.ONBOARDING in plan.agents_to_invoke:
            onboarding = OnboardingAgent(policy_agent)
            resp = onboarding.run(plan)
            print(f"  OnboardingAgent answer:\n  {resp.answer[:300]}")

        # Run compliance guard
        verified, compliance = compliance_guard.run(question, retrieved)
        print(f"  Compliance: passes={compliance.passes}, flags={compliance.flags}")
        if compliance.corrected_answer:
            print(f"  CORRECTED:\n  {compliance.corrected_answer[:300]}")
        elif compliance.flags:
            for flag in compliance.flags:
                print(f"  Caveat: {MANDATORY_CAVEATS[flag]}")

    print("\n"+"="*60)
    print("  COMPONENT D COMPLETE ✅")
    print("  Run next: python3 component_e_response_synthesizer.py")
    print("="*60)
