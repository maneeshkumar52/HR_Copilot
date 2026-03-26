# HR Copilot — Workshop Build Guide
### WS01 · MLDS 2026 · Maneesh Kumar & Ravikiran Ravada

---

## Problem Statement

### The Challenge

Enterprise HR teams spend 60–70% of their time answering **repetitive, policy-level questions** from employees:

> *"How many sick leave days do I have?"*
> *"Can I work from abroad for two weeks?"*
> *"What happens to my L&D budget if I don't use it?"*

A single monolithic LLM cannot reliably answer all of these because:

| Limitation | Why it matters |
|---|---|
| HR spans multiple domains | Leave, Compensation, Remote Work, Compliance, L&D, Onboarding — each has distinct rules |
| Policy docs + structured data | Some answers need PDF/Markdown retrieval; others need exact numbers from CSV/JSON |
| Legal and compliance risk | Wrong advice on POSH, termination, or salary can create legal liability |
| Hallucination | LLMs confidently produce wrong numbers (leave days, salary bands, timelines) |
| No single correct answer | "I want to take leave and also check my increment" spans 2 domains simultaneously |

### The Solution: Multi-Agent HR Copilot

Build a **5-agent pipeline** where each agent has a focused responsibility:

```
Employee Question
      │
      ▼
[A] Knowledge Base     ← index HR policy docs + structured data
      │
      ▼
[B] OrchestratorAgent  ← classify intent, decide which agents to invoke
      │
      ├──► [C1] PolicyRAGAgent      ← hybrid RAG over policy documents
      ├──► [C2] DataQueryAgent      ← exact lookup over CSV/JSON data
      └──► [C3] OnboardingAgent     ← specialist for new joiner questions
                    │
                    ▼
             [D] ComplianceGuardAgent  ← rerank, NLI fact-check, legal scan
                    │
                    ▼
             [E] ResponseSynthesizerAgent  ← merge, format, RAGAS evaluate
                    │
                    ▼
             Final Answer to Employee
```

---

## What You Will Build — 5 Components

| Component | File | What it does |
|---|---|---|
| A — Knowledge Base | `component_a_hr_indexing.py` | Load 6 HR docs, chunk, embed, build FAISS + BM25 index |
| B — Orchestrator | `component_b_orchestrator_agent.py` | Classify intent (9 HR intents), route to correct agents, decompose multi-domain queries |
| C — Specialist Agents | `component_c_policy_data_agents.py` | PolicyRAGAgent (hybrid retrieval), DataQueryAgent (pandas), OnboardingAgent (checklist) |
| D — Compliance Guard | `component_d_compliance_guard.py` | Cross-encoder rerank, NLI fact-check, POSH/legal compliance scan |
| E — Synthesizer + Eval | `component_e_response_synthesizer.py` | Merge all agent outputs, format response, evaluate with RAGAS (Faithfulness, Relevancy, Precision) |

---

## Component A — HR Knowledge Base Indexing

**Goal:** Convert HR policy documents into a searchable index that all agents can query.

### What to build

1. **Document loader** — reads `.md`, `.pdf`, `.docx` files from `data/hr_docs/`
2. **HR-aware chunker** — splits documents at section boundaries (`\n## `, `\n### `, `\n\n`)
   - Chunk size: ~800 characters (2× token estimate) — keeps individual policy clauses intact
   - Overlap: ~120 characters — prevents clause split across chunk boundary
3. **Embedding model** — `all-MiniLM-L6-v2` (384-dim, local, no API key)
4. **FAISS flat index** — `IndexFlatIP` with L2-normalised vectors (inner product = cosine similarity)
5. **BM25 keyword index** — `BM25Okapi` with HR-cleaned tokenisation
6. **Save to disk** — `data/index/hr_faiss.index`, `hr_bm25.pkl`, `hr_chunks.json`

### Key design decisions to explain

- **Why chunk at section boundaries?** HR clauses (e.g., "Carry-forward rule: max 8 days") are self-contained. Splitting mid-sentence loses the rule. Splitting at `## Annual Leave` keeps the full entitlement clause in one chunk.
- **Why save to disk?** All 5 downstream agents share the same index — loading once and saving avoids re-embedding on every run (slow: ~30s).
- **Azure equivalent:** Azure AI Document Intelligence (layout-aware chunking) + Azure AI Search (HNSW, hybrid retrieval, semantic ranker built-in).

### HR category mapping

Each chunk is tagged with its HR domain category. This tag is used by the OrchestratorAgent to filter retrieval to the relevant domain only.

```python
HR_CATEGORY_MAP = {
    "leave_policy":          "leave",
    "compensation_benefits": "compensation",
    "remote_work_policy":    "remote_work",
    "onboarding_guide":      "onboarding",
    "grievance_compliance":  "grievance",
    "learning_development":  "learning",
}
```

### Expected output

```
Total HR documents: 6
Total chunks: 78
  [compensation   ]: 10 chunks
  [grievance      ]: 14 chunks
  [learning       ]: 15 chunks
  [leave          ]: 13 chunks
  [onboarding     ]: 16 chunks
  [remote_work    ]: 10 chunks
```

---

## Component B — OrchestratorAgent (Multi-Agent Router)

**Goal:** Understand the employee's intent and decide which specialist agents to invoke.

### What to build

1. **9 HR intent classes** — `QueryIntent` enum covering every HR domain:
   - `leave_policy`, `compensation`, `remote_work`, `onboarding`, `grievance`, `learning`, `headcount_data`, `salary_band`, `multi_domain`, `unknown`

2. **Rule-based intent classifier** — regex keyword matching per intent (fast, zero LLM cost)
   - Example: `r"\b(leave|sick leave|carry.forward|maternity|paternity)\b"` → `LEAVE_POLICY`
   - When 3+ intents match → `MULTI_DOMAIN` (spans multiple domains)

3. **Query decomposer** — breaks complex questions into independent sub-queries
   - Coordination patterns: `" and also "`, `" as well as "`
   - Conditional patterns: `"If I [X], what happens to [Y]?"` → two separate sub-questions

4. **Routing table** — maps intent → list of agents to invoke
   ```python
   ROUTING_TABLE = {
       QueryIntent.LEAVE_POLICY:  [PolicyRAGAgent, ComplianceGuardAgent],
       QueryIntent.COMPENSATION:  [PolicyRAGAgent, DataQueryAgent, ComplianceGuardAgent],
       QueryIntent.SALARY_BAND:   [DataQueryAgent, ComplianceGuardAgent],
       ...
   }
   ```

5. **Priority docs map** — maps intent → category string for scoped retrieval
   - Must match the `category` field in `hr_chunks.json` (set by Component A)

6. **LLM fallback** — Ollama/Mistral 7B for structured JSON routing plan (optional)

### Key design decisions to explain

- **Why not just use one big LLM?** A Leave question + a Compensation question need different retrieval scopes. Routing to specialists means each agent retrieves from its own filtered domain → higher precision.
- **Why rule-based first?** Keyword matching is deterministic, zero-latency, zero-cost. LLM routing is non-deterministic and slow. Use LLM only when rules don't match confidently.
- **`HRQueryPlan` dataclass** — the typed contract that flows through the entire pipeline. No agent passes raw strings to another. This prevents hallucination contamination across agents.

### `HRQueryPlan` — the typed agent contract

```python
@dataclass
class HRQueryPlan:
    original_question:  str
    intent:             QueryIntent
    sub_queries:        List[str]      # decomposed sub-questions
    agents_to_invoke:   List[AgentName]
    priority_docs:      List[str]      # category filter for retrieval
    needs_structured:   bool           # True → invoke DataQueryAgent
    reasoning:          str
    confidence:         float
```

---

## Component C — Specialist Agents (PolicyRAGAgent + DataQueryAgent)

**Goal:** Answer HR questions using the right retrieval method for each query type.

### PolicyRAGAgent — Hybrid Retrieval

Two retrieval signals, fused together:

| Signal | Method | Strength |
|---|---|---|
| Semantic similarity | FAISS vector search (all-MiniLM-L6-v2) | Finds conceptually related clauses even with different words |
| Keyword match | BM25 (Okapi) | Catches exact HR terms: "carry-forward", "POSH", "Band 4" |

**RRF (Reciprocal Rank Fusion)** — combines both lists without needing a shared score scale:

```
RRF_score(chunk) = Σ  1 / (rank_in_list + 60)
```

- Top 10 from FAISS + Top 8 from BM25 → deduplicate → RRF rank → return Top 6
- Category filter: only retrieve chunks whose `category` matches `plan.priority_docs`

### DataQueryAgent — Structured Data Lookup

Handles queries that need **exact numbers** from:
- `headcount.csv` — employee count by department, attrition rate, band distribution
- `salary_bands.json` — CTC range by band (B1–B6), ESOP eligibility

Uses pandas for exact lookups — does not hallucinate salary numbers.

### OnboardingAgent — Checklist Specialist

Combines two sources:
1. PolicyRAGAgent retrieval scoped to `onboarding` category
2. Hard-coded onboarding checklist (Day 1, Week 1, Pre-joining, 30-60-90 day plan)

Why specialist? New joiners ask procedural questions (`"What do I bring on Day 1?"`). A general agent over-retrieves from unrelated policy domains.

### Key design decisions to explain

- **Why hybrid RAG?** Semantic search misses exact terms. BM25 misses semantic variants. Together they cover both. Example: *"carry-forward limit"* → BM25 finds "carry-forward", semantic finds "unused leave lapses".
- **Why RRF?** FAISS scores (cosine) and BM25 scores (TF-IDF) are not on the same scale. RRF uses rank order only — no normalisation needed.
- **Why separate DataQueryAgent?** Policy docs say "Band 4 gets ESOP" but not the exact CTC range. The JSON/CSV has the exact numbers. Combining both gives complete answers.

---

## Component D — ComplianceGuardAgent

**Goal:** Ensure every retrieved chunk is relevant and every answer is legally safe.

### Three-stage pipeline

```
Retrieved chunks (from PolicyRAGAgent)
        │
        ▼
1. Cross-Encoder Reranking
   model: ms-marco-MiniLM-L6-v2
   Evaluates (query, chunk) JOINTLY → much better relevance than bi-encoder alone
   Returns top 8 chunks above logit threshold
        │
        ▼
2. NLI Fact Check
   model: nli-deberta-v3-small
   Checks: does this chunk ENTAIL (support) the answer to the question?
   Labels: ENTAILMENT / NEUTRAL / CONTRADICTION
   Only ENTAILMENT chunks pass to the Synthesizer
        │
        ▼
3. Legal Compliance Scan
   Scans question + answer for sensitive HR topics:
   → termination, POSH, PIP, legal threats, salary cut
   Adds mandatory caveats (e.g., "Consult your HRBP and Legal team")
   POSH complaints → BLOCKED entirely, redirected to ICC
```

### Why each stage matters (HR-specific)

| Stage | HR Risk it prevents |
|---|---|
| Reranking | Casual Leave chunk retrieved for Annual Leave question (same domain, wrong clause) |
| NLI fact-check | Chunk contains "leave" but answers a different question entirely |
| Compliance scan | Copilot advising directly on POSH complaint → legal liability |

### Sensitive topic handling

```python
SENSITIVE_PATTERNS = {
    "termination": r"\b(terminat|dismiss|fired|sacked|end of employment)\b",
    "posh":        r"\b(posh|sexual harassment|misconduct)\b",
    "pip":         r"\b(pip|performance improvement|underperform)\b",
    "legal":       r"\b(sue|lawsuit|legal action|tribunal|whistleblow)\b",
    "salary_cut":  r"\b(salary cut|pay cut|ctc reduction|demotion)\b",
}
```

- All sensitive topics → add mandatory legal caveat to response
- POSH + "complaint" → block response entirely, redirect to ICC (icc@enterprise.com)

### Key design decisions to explain

- **Bi-encoder vs Cross-encoder:** Component A uses bi-encoder (fast, encodes query and doc independently). Component D uses cross-encoder (slow but accurate, encodes query+doc jointly). Use bi-encoder for retrieval (top 10 from thousands), cross-encoder for reranking (top 8 from 10).
- **Why NLI?** Retrieved chunks share domain terms but may not answer the specific question. NLI entailment check filters "related but irrelevant" chunks before synthesis.

---

## Component E — Response Synthesizer + RAGAS Evaluation

**Goal:** Merge all agent outputs into one clean, employee-facing answer and measure its quality.

### What to build

**1. Response synthesis (two paths):**

- **Ollama/LLM path** — sends verified chunks as context to Mistral 7B with an HR-specific system prompt
- **Template fallback** — keyword-aware extraction from chunks (no LLM required)
  - Finds sections whose headers contain question keywords
  - Converts markdown headers (`## Sick Leave`) to bold text (`**Sick Leave**`) — prevents Streamlit rendering raw H2/H3 headings in the UI
  - Falls back to full chunk text if no keyword match

**2. RAGAS evaluation (3 metrics):**

| Metric | What it measures | How computed |
|---|---|---|
| **Faithfulness** | Is every claim in the answer grounded in retrieved chunks? | NLI entailment: answer sentences vs. chunk sentences |
| **Answer Relevancy** | Does the answer actually address the question? | Cosine similarity: question embedding vs. answer embedding |
| **Context Precision** | Are the retrieved chunks actually relevant? | Answer token overlap with each chunk |

**3. CI/CD gate:**
- Faithfulness ≥ 0.80 → PASS
- Runs as a 6-question eval suite via `python hr_copilot_pipeline.py --eval --gate`

### Key design decisions to explain

- **Why RAGAS?** Unlike manual QA, RAGAS is automated and runs on every change. Faithfulness catches hallucinations. Answer Relevancy catches off-topic answers. Both run locally without needing labelled ground truth.
- **Why template fallback?** The system must work in workshop environments without internet or GPU. Template fallback uses regex + keyword extraction — no LLM, no API key, no Ollama required.

---

## Shared Data Models — `hr_data_models.py`

All 5 components communicate through **typed dataclasses**. Never pass raw strings between agents.

```
HRQueryPlan         ← OrchestratorAgent → all downstream agents
RetrievedChunk      ← PolicyRAGAgent → ComplianceGuardAgent → Synthesizer
AgentResponse       ← each Specialist Agent → Synthesizer
ComplianceCheckResult ← ComplianceGuardAgent → Synthesizer
FinalHRResponse     ← Synthesizer → UI / CLI
```

This is the inter-agent contract. Changing one agent's output format is safe as long as the dataclass interface stays the same.

---

## Build Order and Dependencies

Build in sequence — each component depends on the previous:

```
Component A  →  Index must exist before retrieval can run
     │
Component B  →  OrchestratorAgent produces HRQueryPlan
     │
Component C  →  Reads HRQueryPlan, produces AgentResponse + RetrievedChunk list
     │
Component D  →  Reads RetrievedChunk list, produces verified chunks + ComplianceCheckResult
     │
Component E  →  Reads everything, produces FinalHRResponse + RAGAS scores
     │
Pipeline     →  hr_copilot_pipeline.py wires all 5 components together
     │
UI           →  hr_copilot_ui.py runs the pipeline via Streamlit
```

### Run order for workshop

```bash
# Step 1: one-time setup
chmod +x setup.sh && ./setup.sh

# Step 2: verify each component individually
python component_a_hr_indexing.py           # should print: 78 chunks, test query result
python component_b_orchestrator_agent.py    # should print: 6 routing scenarios with ✅
python component_c_policy_data_agents.py    # should print: retrieved chunks per query
python component_d_compliance_guard.py      # should print: compliance flags + verified chunks
python component_e_response_synthesizer.py  # should print: full answer + RAGAS scores

# Step 3: run full pipeline
python hr_copilot_pipeline.py --eval        # 6-question eval suite

# Step 4: launch UI
streamlit run hr_copilot_ui.py              # http://localhost:8501
```

---

## Knowledge Base — 6 HR Policy Documents

| File | HR Domain | Key topics |
|---|---|---|
| `leave_policy.md` | Leave | Annual (24 days, CF 8 days), Casual (6 days), Sick (12 days), Maternity (26 weeks), Paternity (5 days), Bereavement, Marriage, Comp-Off |
| `compensation_benefits.md` | Compensation | Salary bands B1–B6, ESOP (Band 4+), tax-exempt allowances, variable pay, increment matrix |
| `remote_work_policy.md` | Remote Work | Hybrid 3 days/week (post-probation), WFA rules (30 days max/year), core hours 10–5 IST, approval process |
| `onboarding_guide.md` | Onboarding | Pre-joining checklist, Day 1 checklist, Week 1 mandatory training (POSH, CoC), 30-60-90 day plan |
| `grievance_compliance.md` | Grievance | Grievance channels, POSH policy + ICC process, disciplinary procedure, notice periods by band |
| `learning_development.md` | L&D | Rs 25,000 annual budget, eligible certifications (AWS/Azure/GCP/PMP/CFA), claim process, clawback policy |

---

## Structured Data — 2 Files

| File | Contents | Used by |
|---|---|---|
| `data/hr_structured/salary_bands.json` | CTC min/max by band B1–B6, ESOP %, variable pay % | DataQueryAgent |
| `data/hr_structured/headcount.csv` | Headcount by dept, attrition, open positions, band distribution | DataQueryAgent |

---

## Local vs Azure Production Architecture

| Component | Local (Workshop) | Azure Production |
|---|---|---|
| Documents | `.md` files | Azure Blob Storage / SharePoint |
| Embeddings | `all-MiniLM-L6-v2` (384-dim) | `text-embedding-3-large` (3072-dim) |
| Vector search | FAISS `IndexFlatIP` | Azure AI Search (HNSW, built-in hybrid) |
| Keyword search | BM25Okapi | Azure AI Search (built-in BM25) |
| LLM | Mistral 7B via Ollama | Azure OpenAI GPT-4o |
| Reranker | `ms-marco-MiniLM-L6-v2` (local) | Azure AI Search Semantic Ranker (L2 neural) |
| NLI check | `nli-deberta-v3-small` (local) | Azure AI Language (text classification) |
| Agent routing | Rule-based + Ollama | Azure OpenAI structured output (`response_format=JSON`) |
| Message passing | Python function calls | Azure Service Bus |
| UI | Streamlit (localhost:8501) | Azure Container Apps |
| Evaluation | Local RAGAS | Azure AI Evaluation SDK |

---

## Common Pitfalls and How to Avoid Them

| Pitfall | Cause | Fix |
|---|---|---|
| "No relevant policy found" | Index not built or outdated | Run `python component_a_hr_indexing.py` |
| Wrong intent classified | Keyword not in regex pattern | Add to `INTENT_KEYWORDS` in Component B |
| Category filter returning 0 chunks | `priority_docs` values don't match `category` field in index | Must match exactly: `"leave"`, `"compensation"`, `"remote_work"`, `"onboarding"`, `"grievance"`, `"learning"` |
| All chunks filtered after reranking | Cross-encoder returns negative logits; threshold too high | Set `RERANK_THRESH = -10.0` (keeps all chunks) |
| Markdown headers rendered as H2 in UI | Raw chunk text contains `## Section Name` | Convert to `**bold**` in synthesizer before returning |
| Multiple leave types in one chunk | Chunk size too large (1600 chars) — several short sections bundled | Reduce to 800 chars; each leave type gets its own chunk |
| Segfault loading 3 ML models together | macOS semaphore + multiprocessing conflict | Set `TOKENIZERS_PARALLELISM=false` and `OMP_NUM_THREADS=1` before imports |

---

## 20 Dry-Run Questions (Verified Working)

Use these to validate the system end-to-end. All 20 return correct answers with Faithfulness = 1.00.

### Leave Policy
1. What is the maximum annual leave carry-forward and can it be encashed?
2. How many sick leave days am I entitled to per year?
3. What is the maternity leave duration at Enterprise Corp?
4. Can I combine casual leave and annual leave for a long trip?
5. What bereavement leave am I entitled to if a parent passes away?

### Compensation & Benefits
6. What is the salary band range for a Band 3 Senior Lead?
7. What is the CTC range for a Band 4 Manager and are they eligible for ESOP?
8. Which monthly allowances are tax-exempt and what are their amounts?
9. How is variable pay calculated and when is it paid?
10. What is the performance rating and increment matrix?

### Remote Work
11. Can I work from home 3 days a week as a confirmed employee?
12. Can I work from abroad for a month? What approvals do I need?
13. What are the core working hours I must be available during?
14. I am on probation — how many days must I come to office?
15. How do I apply for a recurring weekly remote work schedule?

### Learning & Development
16. What is my annual L&D budget and which certifications can I claim?
17. How do I claim reimbursement for an AWS certification exam?
18. Does unused L&D budget carry forward to next year?
19. What internal learning programs does Enterprise Corp offer?
20. What is the clawback policy if I leave after a company-sponsored certification?

---

*HR Copilot · WS01 · MLDS 2026 · Maneesh Kumar & Ravikiran Ravada*
