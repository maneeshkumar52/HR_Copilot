"""
============================================================
COMPONENT C — PolicyRAGAgent + DataQueryAgent
============================================================
HR Copilot · MLDS 2026 · Maneesh Kumar & Ravikiran Ravada

Use Case:
  Two specialist agents run in parallel, each expert in its domain:

  PolicyRAGAgent — searches the HR policy document corpus using
    hybrid retrieval (FAISS vector + BM25 keyword + RRF fusion).
    Handles: Leave, Compensation, Remote Work, Grievance, L&D policies.

  DataQueryAgent — queries structured HR data using pandas.
    Handles: headcount.csv, salary_bands.json.
    Returns precise numbers that policy documents can't provide.

Why two agents?
  Unstructured policy docs → best answered by semantic RAG
  Structured data (CSVs, JSONs) → best answered by exact lookup
  Merging both → richer, more accurate answers than either alone.

Azure equivalents:
  PolicyRAGAgent  → Azure AI Search (hybrid) + Azure OpenAI
  DataQueryAgent  → Azure SQL / Cosmos DB + NL-to-SQL agent

Run: python3 component_c_policy_data_agents.py
============================================================
"""
import re, json, math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from hr_data_models import (
    HRQueryPlan, RetrievedChunk, AgentResponse, AgentName, QueryIntent
)
from component_a_hr_indexing import load_index, INDEX_DIR
from component_b_orchestrator_agent import orchestrator_agent

print("="*60)
print("  COMPONENT C — PolicyRAGAgent + DataQueryAgent")
print("="*60)

# Retrieval config — balanced for relevancy (0.7+) AND precision (0.7+)
TOP_K_VEC  = 9       # Vector search candidates (higher for relevancy coverage)
TOP_K_BM25 = 7       # BM25 keyword search candidates
TOP_K_RRF  = 6       # Final merged chunks (increased from 4 for better relevancy)
RRF_K      = 60     # RRF constant — see WS01 Component C explanation


# ═══════════════════════════════════════════════════════════════
# POLICY RAG AGENT
# Hybrid retrieval: FAISS vector + BM25 + RRF fusion
# ═══════════════════════════════════════════════════════════════
class PolicyRAGAgent:
    """
    Specialist agent for HR policy document retrieval.

    Architecture:
      Query → Vector Search (FAISS) ──┐
      Query → BM25 Keyword Search ────┤→ RRF Fusion → Category Filter → Top-K chunks
                                      │
      (Azure: both merged internally by Azure AI Search hybrid mode)
    """

    def __init__(self, chunks: List[Dict], faiss_index: faiss.Index,
                 bm25: BM25Okapi, model: SentenceTransformer):
        self.chunks      = chunks
        self.faiss_index = faiss_index
        self.bm25        = bm25
        self.model       = model
        self.name        = AgentName.POLICY_RAG

    def _vector_search(self, query: str, category_filter: Optional[List[str]],
                       top_k: int = TOP_K_VEC) -> List[Tuple[int, float]]:
        """FAISS cosine similarity search with TWO-PASS category filter.
        Pass 1: Return chunks from priority categories.
        Pass 2: If not enough, backfill with chunks from other categories.
        Balances precision with relevancy."""
        q_vec = self.model.encode([query], normalize_embeddings=True).astype("float32")
        search_k = top_k * 3 if category_filter else top_k
        D, I = self.faiss_index.search(q_vec, search_k)
        
        results_priority = []
        results_fallback = []
        
        for idx, score in zip(I[0], D[0]):
            if idx < 0:
                continue
            chunk_cat = self.chunks[idx]["category"]
            if category_filter and chunk_cat in category_filter:
                results_priority.append((int(idx), float(score)))
            else:
                results_fallback.append((int(idx), float(score)))
        
        # Merge: priority first, then backfill
        merged = results_priority + results_fallback
        return merged[:top_k]

    def _bm25_search(self, query: str, category_filter: Optional[List[str]],
                     top_k: int = TOP_K_BM25) -> List[Tuple[int, float]]:
        """BM25 keyword search with TWO-PASS category filter.
        Pass 1: Return chunks from priority categories.
        Pass 2: If not enough, backfill with chunks from other categories.
        Balances precision with relevancy."""
        tokens = re.sub(r'[^\w\s]', '', query).lower().split()
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        
        results_priority = []
        results_fallback = []
        
        for idx, score in ranked:
            if score <= 0:
                continue
            chunk_cat = self.chunks[idx]["category"]
            if category_filter and chunk_cat in category_filter:
                results_priority.append((int(idx), float(score)))
            else:
                results_fallback.append((int(idx), float(score)))
        
        # Merge: priority first, then backfill
        merged = results_priority + results_fallback
        return merged[:top_k]

    def _rrf_fusion(self, vec_results: List[Tuple[int,float]],
                    bm25_results: List[Tuple[int,float]],
                    k: int = RRF_K) -> List[Tuple[int, float]]:
        """
        Reciprocal Rank Fusion.
        RRF(doc) = Σ 1/(k + rank_in_list)

        WHY NOT SCORE NORMALISATION:
        Vector cosine scores cluster at [0.85–0.97]. BM25 scores range 0–50+.
        Normalising different-scale scores gives wrong weights.
        RRF uses RANKS only — completely scale invariant.

        Both are ranked #1 → combined score = 2/(60+1) ≈ 0.033
        Both are ranked #5 → combined score = 2/(60+5) ≈ 0.031
        The top-ranked chunks from BOTH sources get the biggest boost.
        """
        scores: Dict[int, float] = {}
        for rank, (idx, _) in enumerate(vec_results, 1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
        for rank, (idx, _) in enumerate(bm25_results, 1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def retrieve(self, sub_query: str, plan: HRQueryPlan,
                 top_k: int = TOP_K_RRF) -> List[RetrievedChunk]:
        """
        Run hybrid retrieval for one sub-query.
        Category filter comes from the Orchestrator's priority_docs OR
        intelligently inferred from sub-query for MULTI_DOMAIN questions.
        """
        # For MULTI_DOMAIN questions, infer domain per sub-query to avoid over-retrieval
        if plan.intent.value == "multi_domain":
            cat_filter = self._infer_category_for_subquery(sub_query)
        else:
            cat_filter = plan.priority_docs if plan.priority_docs else None

        vec_res  = self._vector_search(sub_query, cat_filter)
        bm25_res = self._bm25_search(sub_query, cat_filter)
        fused    = self._rrf_fusion(vec_res, bm25_res)

        vec_score_map  = {i: s for i, s in vec_res}
        bm25_score_map = {i: s for i, s in bm25_res}

        results = []
        for idx, rrf_score in fused[:top_k]:
            c = self.chunks[idx]
            results.append(RetrievedChunk(
                chunk_id    = c["chunk_id"],
                text        = c["text"],
                source_file = c["source_file"],
                category    = c["category"],
                rrf_score   = rrf_score,
                agent_source= self.name.value,
            ))
        return results

    def _infer_category_for_subquery(self, sub_question: str) -> List[str]:
        """
        Smart category inference for multi-domain sub-queries.
        Prevents fetching from all documents when we know the specific domain.
        """
        q = sub_question.lower()
        
        # ESOP/Stock options → compensation
        if "esop" in q or "stock" in q:
            return ["compensation"]
        
        # Leave/Carry-forward/Encash → leave
        if "leave" in q or "carry" in q or "encash" in q:
            return ["leave"]
        
        # Notice period → compensation
        if "notice" in q:
            return ["compensation"]
        
        # Salary/Band/CTC → compensation
        if "salary" in q or "band" in q or "ctc" in q:
            return ["compensation"]
        
        # Remote/WFH → remote_work
        if "remote" in q or "wfh" in q or "work from home" in q:
            return ["remote_work"]
        
        # Joining/Onboarding/Day 1 → onboarding
        if "joining" in q or "onboard" in q or "day 1" in q or "first week" in q:
            return ["onboarding"]
        
        # Grievance/Complaint/POSH → grievance
        if "grievance" in q or "complaint" in q or "posh" in q:
            return ["grievance"]
        
        # Training/L&D/Budget → learning
        if "training" in q or "l&d" in q or "learning" in q or "course" in q:
            return ["learning"]
        
        # No match → retrieve from all (fallback)
        return None

    def run(self, plan: HRQueryPlan) -> AgentResponse:
        """
        Execute all sub-queries and merge deduplicated results.

        PARALLEL EXECUTION (production pattern):
            import asyncio
            results = await asyncio.gather(
                *[async_retrieve(sq) for sq in plan.sub_queries]
            )
        Azure: Azure AI Search supports batch requests — multiple queries in one API call.
        """
        print(f"\n  [PolicyRAGAgent] Retrieving for {len(plan.sub_queries)} sub-queries")
        seen: Dict[str, RetrievedChunk] = {}

        for sq in plan.sub_queries:
            chunks = self.retrieve(sq, plan)
            print(f"    SQ: '{sq[:60]}...' → {len(chunks)} chunks")
            for c in chunks:
                if c.chunk_id not in seen:
                    seen[c.chunk_id] = c
                else:
                    seen[c.chunk_id].rrf_score += c.rrf_score * 0.5   # boost cross-query

        deduped = sorted(seen.values(), key=lambda x: x.rrf_score, reverse=True)[:8]
        print(f"    Deduplicated: {len(deduped)} unique chunks")

        # Format a raw answer (will be refined by Component E's Synthesizer)
        answer_parts = []
        for i, c in enumerate(deduped[:4], 1):
            answer_parts.append(f"[Source {i}: {c.source_file.split('/')[-1]}]\n{c.text[:300]}")
        raw_answer = "\n\n".join(answer_parts) if answer_parts else "No relevant policy found."

        return AgentResponse(
            agent       = self.name,
            answer      = raw_answer,
            sources     = list({c.source_file.split("/")[-1] for c in deduped}),
            confidence  = deduped[0].rrf_score if deduped else 0.0,
            chunks_used = len(deduped),
        )

    @property
    def retrieved_chunks(self) -> List[RetrievedChunk]:
        """Expose last retrieval result for downstream agents."""
        return getattr(self, '_last_chunks', [])


# ═══════════════════════════════════════════════════════════════
# DATA QUERY AGENT
# Answers structured data questions from CSV / JSON files
# ═══════════════════════════════════════════════════════════════
class DataQueryAgent:
    """
    Specialist agent for structured HR data queries.
    Uses pandas for CSV and json for structured data.

    Why a separate agent?
      Policy RAG retrieves text passages — it cannot compute
      "total headcount" or "salary band range for B4".
      Structured data queries need exact lookup, not semantic search.

    Azure production equivalents:
      - Azure SQL Database with NL-to-SQL agent (Text-to-SQL pattern)
      - Azure Cosmos DB with SQL API queries
      - Azure Synapse Analytics for larger datasets
    """

    DATA_DIR = "data/hr_structured"

    def __init__(self):
        self.name = AgentName.DATA_QUERY
        self._salary_bands = None
        self._headcount    = None

    def _load_salary_bands(self) -> dict:
        if self._salary_bands is None:
            import json
            try:
                with open(f"{self.DATA_DIR}/salary_bands.json") as f:
                    self._salary_bands = json.load(f)
            except FileNotFoundError:
                self._salary_bands = {"bands": []}
        return self._salary_bands

    def _load_headcount(self):
        if self._headcount is None:
            try:
                import pandas as pd
                self._headcount = pd.read_csv(f"{self.DATA_DIR}/headcount.csv")
            except (FileNotFoundError, ImportError):
                self._headcount = None
        return self._headcount

    def query_salary_bands(self, question: str) -> str:
        """Look up salary range for a given band or title."""
        data = self._load_salary_bands()
        bands = data.get("bands", [])
        if not bands:
            return "Salary band data not available.\n\n*Source: salary_bands.json*"

        # Extract band number from question
        band_match = re.search(r'b(\d)', question, re.IGNORECASE)
        title_kw   = re.search(r'(manager|director|associate|lead|vp|senior)', question, re.IGNORECASE)

        results = []
        for band in bands:
            if band_match and f"B{band_match.group(1)}" == band["band"]:
                results.append(band)
            elif title_kw and title_kw.group(1).lower() in band["title"].lower():
                results.append(band)

        if not results:
            # Return all bands as a table
            lines = ["Salary Band Reference:"]
            for b in bands:
                esop = " (ESOP eligible)" if b.get("esop_eligible") else ""
                lines.append(
                    f"  {b['band']} — {b['title']}: "
                    f"₹{b['min_ctc']//100000:.1f}L – ₹{b['max_ctc']//100000:.1f}L CTC{esop}"
                )
            return "\n".join(lines) + "\n\n*Source: salary_bands.json*"

        lines = []
        for b in results:
            esop = " · ESOP eligible" if b.get("esop_eligible") else ""
            notice = b.get("notice_period_months", 2)
            lines.append(
                f"Band {b['band']} ({b['title']}):\n"
                f"  CTC Range:     ₹{b['min_ctc']//100000:.1f}L – ₹{b['max_ctc']//100000:.1f}L\n"
                f"  Notice Period: {notice} months{esop}"
            )
        return "\n\n".join(lines) + "\n\n*Source: salary_bands.json*"

    def query_headcount(self, question: str) -> str:
        """
        Answer headcount, attrition, or vacancy questions using pandas.

        Azure SQL equivalent:
          SELECT department, headcount, attrition_ytd
          FROM hr_headcount
          WHERE department = 'Engineering'
        """
        df = self._load_headcount()
        if df is None:
            return "Headcount data not available. Ensure pandas is installed.\n\n*Source: headcount.csv*"

        q = question.lower()

        # Department-specific query
        dept_match = re.search(
            r'(engineering|product|sales|hr|finance|operations|marketing|legal)',
            q, re.IGNORECASE
        )
        if dept_match:
            dept = dept_match.group(1).title()
            row = df[df['department'].str.lower() == dept.lower()]
            if not row.empty:
                r = row.iloc[0]
                return (
                    f"Headcount — {r['department']}:\n"
                    f"  Current Headcount:  {r['headcount']}\n"
                    f"  Open Positions:     {r['open_positions']}\n"
                    f"  Attrition YTD:      {r['attrition_ytd']}\n"
                    f"  Avg Tenure:         {r['avg_tenure_years']} years\n\n"
                    f"*Source: headcount.csv*"
                )

        # Company-wide summary
        if any(kw in q for kw in ["total", "company", "overall", "all", "how many employees"]):
            total_hc   = df['headcount'].sum()
            total_open = df['open_positions'].sum()
            total_attr = df['attrition_ytd'].sum()
            attr_rate  = round(total_attr / total_hc * 100, 1)
            return (
                f"Company-wide Headcount Summary:\n"
                f"  Total Employees:    {total_hc}\n"
                f"  Open Positions:     {total_open}\n"
                f"  Attrition YTD:      {total_attr} ({attr_rate}%)\n\n"
                f"By Department:\n"
                + "\n".join(
                    f"  {r['department']:20s}: {r['headcount']} employees, {r['open_positions']} open"
                    for _, r in df.iterrows()
                )
                + "\n\n*Source: headcount.csv*"
            )

        # Attrition specific
        if "attrition" in q or "turnover" in q:
            df_sorted = df.sort_values('attrition_ytd', ascending=False)
            return (
                "Attrition YTD by Department:\n"
                + "\n".join(
                    f"  {r['department']:20s}: {r['attrition_ytd']} ({round(r['attrition_ytd']/r['headcount']*100,1)}%)"
                    for _, r in df_sorted.iterrows()
                )
                + "\n\n*Source: headcount.csv*"
            )

        # Fallback: full table
        return self.query_headcount("total company headcount")

    def run(self, plan: HRQueryPlan) -> AgentResponse:
        """Route the data query to the appropriate structured data source."""
        print(f"\n  [DataQueryAgent] Query: '{plan.original_question[:60]}...'")
        answers = []
        sources = []

        for sq in plan.sub_queries:
            sq_lower = sq.lower()
            if any(kw in sq_lower for kw in ["salary", "ctc", "band", "pay", "earn", "compensation range"]):
                answer = self.query_salary_bands(sq)
                sources.append("salary_bands.json")
            elif any(kw in sq_lower for kw in ["headcount", "employees", "team size", "attrition", "vacancy", "open position"]):
                answer = self.query_headcount(sq)
                sources.append("headcount.csv")
            else:
                # Try both
                sb = self.query_salary_bands(sq)
                hc = self.query_headcount(sq)
                answer = sb if "Band" in sb else hc
                sources.extend(["salary_bands.json", "headcount.csv"])

            answers.append(answer)
            print(f"    Answer preview: {answer[:100]}...")

        combined = "\n\n".join(answers)
        return AgentResponse(
            agent       = self.name,
            answer      = combined,
            sources     = list(set(sources)),
            confidence  = 0.92,   # structured data is highly reliable
            chunks_used = 0,
            data_used   = True,
        )


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  Loading indexes...")
    chunks, faiss_idx, bm25, model = load_index(INDEX_DIR)

    policy_agent = PolicyRAGAgent(chunks, faiss_idx, bm25, model)
    data_agent   = DataQueryAgent()

    test_cases = [
        "What is the annual leave carry-forward limit and encashment policy?",
        "What is the salary band range for a Band 4 manager?",
        "What is the total company headcount and attrition rate?",
        "I am joining next week — what documents do I need?",
        "What is the remote work policy for employees on probation?",
    ]

    for question in test_cases:
        print(f"\n{'─'*60}")
        print(f"  Q: {question}")

        plan = orchestrator_agent(question, use_llm=False)

        responses = []
        if AgentName.POLICY_RAG in plan.agents_to_invoke:
            resp = policy_agent.run(plan)
            responses.append(resp)
            print(f"\n  PolicyRAGAgent: {resp.chunks_used} chunks, sources={resp.sources}")

        if AgentName.DATA_QUERY in plan.agents_to_invoke or plan.needs_structured:
            resp = data_agent.run(plan)
            responses.append(resp)
            print(f"\n  DataQueryAgent: data_used={resp.data_used}")
            print(f"  {resp.answer[:200]}")

    # Save for Component D
    import pickle, os
    os.makedirs(INDEX_DIR, exist_ok=True)
    demo_q = "What is the carry-forward limit for annual leave?"
    demo_plan = orchestrator_agent(demo_q, use_llm=False)
    demo_resp = policy_agent.run(demo_plan)

    # Expose chunks for downstream
    seen = {}
    for sq in demo_plan.sub_queries:
        for c in policy_agent.retrieve(sq, demo_plan):
            seen[c.chunk_id] = c
    demo_chunks = list(seen.values())

    with open(f"{INDEX_DIR}/demo_policy_chunks.pkl", "wb") as f:
        pickle.dump({"plan": demo_plan, "chunks": demo_chunks, "resp": demo_resp}, f)
    print(f"\n  Saved demo chunks to {INDEX_DIR}/demo_policy_chunks.pkl")

    print("\n"+"="*60)
    print("  COMPONENT C COMPLETE ✅")
    print("  Run next: python3 component_d_compliance_guard.py")
    print("="*60)
