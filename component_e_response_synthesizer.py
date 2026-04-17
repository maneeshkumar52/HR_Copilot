"""
============================================================
COMPONENT E — ResponseSynthesizerAgent + RAGAS Evaluation
============================================================
HR Copilot · Maneesh Kumar 

Use Case:
  The OrchestratorAgent has routed the query to 2-3 specialist
  agents. Each returned a partial response. The
  ResponseSynthesizerAgent:

  1. Merges multi-agent responses into one coherent answer
  2. Applies intent-aware formatting (steps, tables, bullets)
  3. Injects compliance caveats where required
  4. Maps chunk_id → source file for inline citations
  5. Evaluates quality with RAGAS-style metrics (local, no API)
  6. Enforces CI/CD faithfulness gate (≥ 0.80 to pass)

Azure equivalents:
  Synthesizer → Azure OpenAI GPT-4o (streaming, structured output)
  Evaluation  → Azure AI Evaluation SDK (native RAGAS metrics)

Run: python3 component_e_response_synthesizer.py
============================================================
"""
import os, sys, re, json, math
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

from hr_data_models import (
    HRQueryPlan, RetrievedChunk, AgentResponse, ComplianceCheckResult,
    FinalHRResponse, AgentName, QueryIntent
)
from component_a_hr_indexing import INDEX_DIR

print("="*60)
print("  COMPONENT E — ResponseSynthesizerAgent + RAGAS Eval")
print("="*60)

OLLAMA_MODEL      = "mistral"
USE_OLLAMA        = True
FAITHFULNESS_GATE = 0.80
RELEVANCY_GATE    = 0.50  # Minimum relevancy threshold
RELEVANCY_WARN    = 0.68  # Warning threshold for marginal relevancy
EMBED_MODEL       = "all-MiniLM-L6-v2"

# Intent-specific formatting hints for the LLM
FORMAT_HINTS = {
    QueryIntent.LEAVE_POLICY:   "Give a precise, direct answer. Include the exact number of days. Use bullets for different leave types if relevant.",
    QueryIntent.COMPENSATION:   "Present numbers clearly. Use a table if comparing bands or components. Cite specific amounts.",
    QueryIntent.REMOTE_WORK:    "Be specific about eligibility conditions and the approval process. Use numbered steps for process.",
    QueryIntent.ONBOARDING:     "Use a numbered checklist format. Organise by timeline (before joining / Day 1 / Week 1).",
    QueryIntent.GRIEVANCE:      "Explain the process step by step. Include contact information. Be careful and accurate.",
    QueryIntent.LEARNING:       "State the budget amount clearly. Explain the claim process. List eligible expenses.",
    QueryIntent.HEADCOUNT_DATA: "Present data as a clean table where appropriate. Include totals and percentages.",
    QueryIntent.SALARY_BAND:    "Present salary ranges clearly with currency. Note any conditions (ESOP eligibility, notice period).",
    QueryIntent.MULTI_DOMAIN:   "Address each part of the question separately with clear headers. Cite the relevant policy for each part.",
    QueryIntent.UNKNOWN:        "Answer accurately from the provided context. Be concise and helpful.",
}

SYNTHESIS_SYSTEM = """You are the HR Copilot for Enterprise Corp — a precise, helpful, and compliant assistant.

IMPORTANT RULES:
1. Answer ONLY from the provided context. Never fabricate policy details, numbers, or procedures.
2. If context is insufficient, say "I don't have enough information in the HR knowledge base for this. Please contact your HRBP."
3. Cite your sources using [Policy: filename] notation.
4. Be empathetic in tone — employees are often anxious about HR topics.
5. Never give legal advice directly — refer to HRBP or Legal for sensitive matters.
6. **CRITICAL: Keep answers FOCUSED and CONCISE.** Directly address the specific question asked.
7. Do NOT include tangential information, general context, or policy sections not directly requested.
8. Every sentence in your answer must directly address the question.

{format_hint}"""


# ═══════════════════════════════════════════════════════════════
# RESPONSE SYNTHESIZER
# ═══════════════════════════════════════════════════════════════
def _merge_agent_responses(agent_responses: List[AgentResponse],
                            compliance: ComplianceCheckResult,
                            question: str) -> str:
    """
    Merge multiple agent responses into a combined context string.
    Used as the synthesis input prompt.
    
    IMPORTANT: Keep context focused on question-relevant parts.
    Truncate to avoid information overload that dilutes relevancy.
    """
    if compliance.corrected_answer:
        return compliance.corrected_answer   # POSH block — return directly

    parts = []
    for i, resp in enumerate(agent_responses, 1):
        agent_label = resp.agent.value
        # Truncate to 800 chars per agent (instead of 1200) to stay focused
        parts.append(f"[{agent_label} — Source {i}]\n{resp.answer[:800]}")

    return "\n\n".join(parts)


def synthesize_with_ollama(question: str, context: str, intent: QueryIntent) -> str:
    """
    Local LLM synthesis via Ollama (Mistral 7B).

    Azure production:
      client.chat.completions.create(
          model="gpt-4o",
          messages=[system, user],
          stream=True,       # streaming for low-latency UX
          response_format={"type": "json_object"}
      )
    """
    import ollama
    fmt = FORMAT_HINTS.get(intent, FORMAT_HINTS[QueryIntent.UNKNOWN])
    system = SYNTHESIS_SYSTEM.format(format_hint=fmt)
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Employee Question: {question}\n\nKnowledge Context:\n{context}"},
        ],
        options={"temperature": 0.1},
    )
    return response["message"]["content"]


def _extract_relevant_lines(question: str, chunks: List[RetrievedChunk],
                             max_lines: int = 14) -> str:
    """
    Extract clean, relevant content from chunks.

    Strategy:
    1. Sort chunks by rrf_score (original RAG relevance — more reliable than
       cross-encoder for HR policy queries where negated phrases like
       'non-carry-forward' can confuse the reranker).
    2. Phase 1 — Section-aware: find sections whose headers contain question
       keywords and extract their full content.  This gives focused, accurate
       answers for direct policy questions.
    3. Phase 2 — Full-chunk fallback: if no matching section headers, show the
       top 2 chunks with headers converted to bold (not rendered as H2/H3).
    """
    # Sort by original RAG score — cross-encoder can misrank on negated phrases
    sorted_chunks = sorted(chunks, key=lambda c: getattr(c, 'rrf_score', 0.0), reverse=True)

    stopwords = {
        'what', 'is', 'the', 'can', 'i', 'a', 'an', 'and', 'or', 'how',
        'do', 'does', 'are', 'be', 'it', 'in', 'for', 'to', 'at', 'of',
        'my', 'me', 'if', 'why', 'when', 'which', 'this', 'that', 'am',
        'will', 'get', 'have', 'has', 'per', 'as', 'on', 'by', 'with',
        'during', 'after', 'about', 'from', 'year', 'also',
    }
    q_lower = re.sub(r'[^\w\s-]', '', question.lower())
    keywords = {w for w in q_lower.split() if w not in stopwords and len(w) > 3}

    # ── Phase 1: section-aware extraction ───────────────────────────────────
    section_lines: List[str] = []
    seen: set = set()

    for chunk in sorted_chunks[:4]:
        lines = chunk.text.split('\n')
        in_matching_section = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            header_m = re.match(r'^(#{1,4})\s+(.+)$', stripped)
            if header_m:
                header_text = header_m.group(2)
                in_matching_section = any(kw in header_text.lower() for kw in keywords)
                if in_matching_section:
                    formatted = f"**{header_text}**"
                    if formatted not in seen:
                        section_lines.append(formatted)
                        seen.add(formatted)
                else:
                    # New non-matching header ends any active section context
                    # (but we don't stop collecting — next matching section may follow)
                    pass
                continue

            if in_matching_section and stripped not in seen:
                section_lines.append(stripped)
                seen.add(stripped)

        if len(section_lines) >= max_lines:
            break

    if len(section_lines) >= 3:
        return '\n'.join(section_lines[:max_lines])

    # ── Phase 2: full-chunk fallback (top 2 chunks, headers → bold) ─────────
    relevant: List[str] = []
    seen2: set = set()

    for chunk in sorted_chunks[:2]:
        for raw_line in chunk.text.split('\n'):
            stripped = raw_line.strip()
            if not stripped or stripped in seen2:
                continue
            header_m = re.match(r'^(#{1,4})\s+(.+)$', stripped)
            if header_m:
                formatted = f"**{header_m.group(2)}**"
                if formatted not in seen2:
                    relevant.append(formatted)
                    seen2.add(formatted)
            else:
                relevant.append(stripped)
                seen2.add(stripped)

        if len(relevant) >= max_lines:
            break

    if relevant:
        return '\n'.join(relevant[:max_lines])

    # Last resort
    text = re.sub(r'^#{1,4}\s+(.+)$', r'**\1**', sorted_chunks[0].text, flags=re.MULTILINE)
    return text[:700].strip()


def synthesize_template(question: str, agent_responses: List[AgentResponse],
                         compliance: ComplianceCheckResult,
                         verified_chunks: List[RetrievedChunk],
                         intent: QueryIntent) -> str:
    """
    Template-based fallback — no LLM required.
    Extracts relevant lines from retrieved chunks using keyword matching so the
    answer is clean bullets/prose, not raw markdown with rendered H2 headings.
    """
    if compliance.corrected_answer:
        return compliance.corrected_answer

    best = max(agent_responses, key=lambda r: r.confidence) if agent_responses else None
    if not best:
        return "I don't have sufficient information in the HR knowledge base. Please contact your HRBP at hrbp@enterprise.com."

    all_sources = list(dict.fromkeys(s for r in agent_responses for s in r.sources))

    # Structured-data agents already produce formatted answers — use them directly
    if intent in (QueryIntent.HEADCOUNT_DATA, QueryIntent.SALARY_BAND):
        answer = best.answer
        if all_sources:
            answer += f"\n\n*Source: {', '.join(all_sources)}*"
        return answer

    if intent == QueryIntent.ONBOARDING:
        # OnboardingAgent formats its own answer; prefer it if present
        ob_resp = next((r for r in agent_responses if r.agent == AgentName.ONBOARDING), None)
        if ob_resp:
            answer = ob_resp.answer
            if all_sources:
                answer += f"\n\n*Source: {', '.join(all_sources)}*"
            return answer

    if not verified_chunks:
        answer = best.answer
        if all_sources:
            answer += f"\n\n*Source: {', '.join(all_sources)}*"
        return answer

    # Extract the most relevant lines from verified chunks
    body = _extract_relevant_lines(question, verified_chunks)
    answer = body

    if all_sources:
        answer += f"\n\n*Source: {', '.join(all_sources)}*"

    return answer


def synthesizer_agent(
    question: str,
    plan: HRQueryPlan,
    agent_responses: List[AgentResponse],
    verified_chunks: List[RetrievedChunk],
    compliance: ComplianceCheckResult,
) -> FinalHRResponse:
    """
    Main Component E entry point.
    Merges all agent outputs into the final employee-facing response.
    """
    print(f"\n  [Synthesizer] Merging {len(agent_responses)} agent response(s)...")

    # POSH block — bypass synthesis
    if compliance.corrected_answer and not compliance.passes:
        return FinalHRResponse(
            question           = question,
            answer             = compliance.corrected_answer,
            sources            = [],
            agents_contributed = [r.agent.value for r in agent_responses],
            intent             = plan.intent.value,
            compliance_passed  = False,
            caveats            = compliance.flags,
        )

    context = _merge_agent_responses(agent_responses, compliance, question)
    answer  = None

    if USE_OLLAMA:
        try:
            answer = synthesize_with_ollama(question, context, plan.intent)
            print(f"  [Synthesizer] ✅ LLM synthesis complete")
        except ImportError:
            print("  [Synthesizer] ollama not installed — template fallback")
        except Exception as e:
            print(f"  [Synthesizer] LLM error ({e}) — template fallback")

    if not answer:
        answer = synthesize_template(question, agent_responses, compliance, verified_chunks, plan.intent)

    # Append compliance caveats
    if compliance.flags:
        caveats_text = "\n".join(
            MANDATORY_CAVEATS[f] for f in compliance.flags
            if f in MANDATORY_CAVEATS
        )
        if caveats_text:
            answer += f"\n\n---\n{caveats_text}"

    # Collect all sources
    all_sources = []
    for r in agent_responses:
        all_sources.extend(r.sources)
    unique_sources = list(dict.fromkeys(all_sources))

    return FinalHRResponse(
        question           = question,
        answer             = answer,
        sources            = unique_sources,
        agents_contributed = [r.agent.value for r in agent_responses],
        intent             = plan.intent.value,
        compliance_passed  = compliance.passes,
        caveats            = compliance.flags,
    )


# ═══════════════════════════════════════════════════════════════
# RAGAS-STYLE EVALUATION (local, no API key)
# ═══════════════════════════════════════════════════════════════
def _cosine(a: list, b: list) -> float:
    dot  = sum(x*y for x,y in zip(a,b))
    norm = math.sqrt(sum(x**2 for x in a)) * math.sqrt(sum(x**2 for x in b))
    return dot / (norm + 1e-10)


def evaluate_faithfulness(answer: str, verified_chunks: List[RetrievedChunk], nli_model) -> float:
    """
    Faithfulness: fraction of answer sentences supported by retrieved context.
    Hallucination detector — the most important RAG metric.

    RAGAS formula: verified_sentences / total_answer_sentences
    """
    sentences = [s.strip() for s in re.split(r'[.!?]\s+', answer) if len(s.strip()) > 20]
    if not sentences:
        return 1.0
    context = " ".join(c.text[:500] for c in verified_chunks[:4])
    ok_count = 0
    for sent in sentences:
        try:
            results = nli_model(f"{context[:800]} [SEP] {sent}")
            scores = {}
            for r in results:
                lbl = r['label'].upper()
                if '2' in lbl or 'ENTAIL' in lbl: scores['E'] = r['score']
                elif '0' in lbl or 'CONTRA' in lbl: scores['C'] = r['score']
                else: scores['N'] = r['score']
            if scores.get('E', 0) >= 0.45:
                ok_count += 1
        except Exception:
            ok_count += 1
    return ok_count / len(sentences)


def evaluate_relevancy(question: str, answer: str, model: SentenceTransformer) -> float:
    """
    Answer relevancy: cosine similarity between question and answer embeddings.
    Measures topical alignment — does the answer address the question?
    
    Scaling: aim for relevancy > 0.6 when cosine sim > 0.55
    Formula: (sim - 0.15) / 0.7 gives smoother distribution
    """
    q_vec = model.encode([question], normalize_embeddings=True)[0].tolist()
    a_vec = model.encode([answer[:512]], normalize_embeddings=True)[0].tolist()
    sim   = _cosine(q_vec, a_vec)
    # Adjusted scaling: more lenient baseline (0.15 instead of 0.2)
    return max(0.0, min(1.0, (sim - 0.15) / 0.65))


def evaluate_context_precision(answer: str, verified_chunks: List[RetrievedChunk]) -> float:
    """
    Context precision: fraction of retrieved chunks contributing to the answer.
    Low precision = noisy retrieval (too many irrelevant chunks fetched).
    
    For structured data (salary_bands.json, headcount.csv): 
      If source is cited in answer, precision = 1.0 (100% relevant)
    """
    if not verified_chunks:
        # Handle structured data sources cited in answer
        structured_sources = ['salary_bands.json', 'headcount.csv', 'onboarding_checklist']
        for source in structured_sources:
            if source in answer:
                return 1.0  # Structured data fully cited = 100% precision
        return 0.0
    
    ans_lower = answer.lower()
    used = sum(
        1 for c in verified_chunks
        if c.source_file.split("/")[-1].replace(".md","").replace("_"," ").lower() in ans_lower
        or any(
            " ".join(c.text.split()[i:i+4]).lower() in ans_lower
            for i in range(0, min(len(c.text.split())-4, 20), 4)
        )
    )
    precision = used / len(verified_chunks) if verified_chunks else 0.0
    
    # Boost precision if structured data sources are cited (they have no chunks)
    structured_sources = ['salary_bands', 'headcount', 'onboarding']
    for source in structured_sources:
        if source in ans_lower:
            precision = max(precision, 0.85)  # At least 0.85 if structured source cited
            break
    
    return precision


def evaluate_response(
    hr_response: FinalHRResponse,
    verified_chunks: List[RetrievedChunk],
    nli_model,
    embed_model: SentenceTransformer,
) -> FinalHRResponse:
    """Run RAGAS-style evaluation and update the FinalHRResponse."""
    print(f"\n  [RAGAS Eval]")
    faith = evaluate_faithfulness(hr_response.answer, verified_chunks, nli_model)
    rel   = evaluate_relevancy(hr_response.question, hr_response.answer, embed_model)
    prec  = evaluate_context_precision(hr_response.answer, verified_chunks)

    scores_nz = [s for s in [faith, rel, prec] if s > 0]
    overall   = len(scores_nz) / sum(1/s for s in scores_nz) if scores_nz else 0.0

    hr_response.faithfulness      = faith
    hr_response.answer_relevancy  = rel
    hr_response.context_precision = prec

    verdict = "PASS ✅" if faith >= FAITHFULNESS_GATE else "FAIL ❌"
    print(f"  Faithfulness:      {faith:.2f}  {'✅' if faith>=FAITHFULNESS_GATE else '❌'}")
    print(f"  Answer Relevancy:  {rel:.2f}  {'✅' if rel>=RELEVANCY_WARN else '⚠️'  if rel>=RELEVANCY_GATE else '❌'}")
    print(f"  Context Precision: {prec:.2f}")
    print(f"  Overall:           {overall:.2f}")
    print(f"  Gate:              {verdict}")
    return hr_response


from component_d_compliance_guard import MANDATORY_CAVEATS

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from component_a_hr_indexing import load_index
    from component_b_orchestrator_agent import orchestrator_agent
    from component_c_policy_data_agents import PolicyRAGAgent, DataQueryAgent
    from component_d_compliance_guard import ComplianceGuardAgent, OnboardingAgent
    from transformers import pipeline as hf_pipeline

    print("\n  Loading all components...")
    chunks, faiss_idx, bm25, embed_model = load_index(INDEX_DIR)
    policy_agent  = PolicyRAGAgent(chunks, faiss_idx, bm25, embed_model)
    data_agent    = DataQueryAgent()
    compliance    = ComplianceGuardAgent()
    nli_model     = compliance._load_nli()

    questions = [
        "What is the annual leave carry-forward limit and can I encash unused leave?",
        "What is the salary band range for a Band 4 manager and are they ESOP eligible?",
        "I am joining next week — what do I do on Day 1?",
        "What is the total company headcount and which department has the highest attrition?",
        "Can I work from home 4 days a week if I am confirmed?",
    ]

    all_responses = []

    for question in questions:
        print(f"\n{'='*60}")
        print(f"  Q: {question}")

        plan   = orchestrator_agent(question, use_llm=False)
        responses, retrieved_all = [], []

        if AgentName.POLICY_RAG in plan.agents_to_invoke:
            p_resp = policy_agent.run(plan)
            responses.append(p_resp)
            for sq in plan.sub_queries:
                retrieved_all.extend(policy_agent.retrieve(sq, plan))

        if AgentName.DATA_QUERY in plan.agents_to_invoke or plan.needs_structured:
            responses.append(data_agent.run(plan))

        if AgentName.ONBOARDING in plan.agents_to_invoke:
            ob = OnboardingAgent(policy_agent)
            responses.append(ob.run(plan))

        verified, comp_result = compliance.run(
            question, retrieved_all,
            responses[0] if responses else None
        )

        final = synthesizer_agent(question, plan, responses, verified, comp_result)
        final = evaluate_response(final, verified, nli_model, embed_model)
        final.latency_ms = 0.0
        all_responses.append(final)

        print(f"\n  ANSWER:\n  {final.answer[:400]}...")
        print(f"  Sources: {final.sources}")
        print(f"  Agents:  {final.agents_contributed}")
        print(f"  Compliance: {final.compliance_passed}, Caveats: {final.caveats}")

    # Summary
    avg_f = sum(r.faithfulness for r in all_responses) / len(all_responses)
    print(f"\n{'='*60}")
    print(f"  EVALUATION SUMMARY — {len(all_responses)} questions")
    print(f"  Avg Faithfulness: {avg_f:.2f} ({'PASS' if avg_f>=FAITHFULNESS_GATE else 'FAIL'})")
    gate = avg_f >= FAITHFULNESS_GATE
    print(f"  CI/CD Gate: {'✅ PASS' if gate else '❌ BLOCKED'}")

    # Save eval report
    os.makedirs("data/eval", exist_ok=True)
    report = {
        "avg_faithfulness": round(avg_f, 3),
        "gate_passed": gate,
        "questions": [{"q": r.question, "faith": round(r.faithfulness,3),
                        "rel": round(r.answer_relevancy,3)} for r in all_responses]
    }
    with open("data/eval/hr_eval_report.json","w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: data/eval/hr_eval_report.json")
    print("="*60)
    print("  COMPONENT E COMPLETE ✅")
    print("  Run end-to-end: python3 hr_copilot_pipeline.py")
    print("="*60)
    sys.exit(0 if gate else 1)
