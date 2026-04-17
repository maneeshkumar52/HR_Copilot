# HR Copilot — Multi-Agent Employee Self-Service System

> **Maneesh Kumar**
> A production-grade multi-agent AI system that answers employee HR questions
> using hybrid retrieval, parallel agent execution, NLI fact-checking, and RAGAS evaluation.

---

## What Is a Multi-Agent System?

A **Multi-Agent System (MAS)** is a collection of specialised software agents
that each handle a narrow domain and collaborate to solve problems no single
agent could handle well alone.

**Why not just use one big LLM?**

| One Monolithic LLM | Multi-Agent System |
|---|---|
| Answers everything from one model | Each agent is a domain expert |
| Accuracy degrades across 9+ HR topics | Specialist accuracy per domain |
| Hard to add legal/compliance gates | Compliance is a dedicated stage |
| No structured data queries | DataQueryAgent handles CSV/JSON |
| No quality measurement | RAGAS evaluation per response |

This project implements a **5-component, 5-agent pipeline** where each
component maps to a distinct AI engineering concept.

---

## Quick Start — 3 Commands

### Mac / Linux
```bash
chmod +x setup.sh && ./setup.sh    # one-time setup (~5 min)
source .venv/bin/activate
streamlit run hr_copilot_ui.py      # opens http://localhost:8501
```

### Windows
```bat
.\setup.bat
.venv\Scripts\activate
streamlit run hr_copilot_ui.py
```

---

## Project Structure

```
HR_Copilot/
│
├── agent_framework.py                ← NEW: Multi-agent framework core
│   ├── BaseAgent                     ← Abstract base every agent extends
│   ├── PlannerAgent                  ← Base for routing/planning agents
│   ├── AgentRegistry                 ← Central service discovery
│   ├── AgentMessage                  ← Typed inter-agent messages
│   ├── AgentTask                     ← Per-agent execution tracker
│   └── ParallelAgentExecutor         ← Runs agents concurrently
│
├── hr_data_models.py                 ← Shared typed dataclasses (contracts)
│   ├── QueryIntent (Enum)            ← 9 HR intent categories
│   ├── AgentName (Enum)              ← Named agents
│   ├── HRQueryPlan                   ← Orchestrator → all agents
│   ├── RetrievedChunk                ← One knowledge chunk
│   ├── AgentResponse                 ← Specialist agent output
│   ├── ComplianceCheckResult         ← Compliance gate output
│   └── FinalHRResponse               ← What the employee sees
│
├── component_a_hr_indexing.py        ← Phase 1: Knowledge Base
├── component_b_orchestrator_agent.py ← Phase 2: Intent Routing
├── component_c_policy_data_agents.py ← Phase 3: Retrieval Agents
├── component_d_compliance_guard.py   ← Phase 4: Compliance Gate
├── component_e_response_synthesizer.py ← Phase 5: Synthesis + Eval
│
├── hr_copilot_pipeline.py            ← CLI entry point (full pipeline)
├── hr_copilot_ui.py                  ← Streamlit web UI
│
├── data/
│   ├── hr_docs/          ← 6 HR policy markdown files
│   ├── hr_structured/    ← salary_bands.json + headcount.csv
│   ├── index/            ← FAISS + BM25 (built by Component A)
│   └── eval/             ← RAGAS evaluation reports
│
├── requirements.txt
├── setup.sh / setup.bat
└── .env.example
```

---

## The Multi-Agent Framework (`agent_framework.py`)

This is the **architectural backbone** — entirely independent of HR logic.

### 1. `BaseAgent` — Uniform Interface

```python
class BaseAgent(ABC):
    @property
    @abstractmethod
    def agent_name(self) -> AgentName: ...

    @abstractmethod
    def run(self, plan: HRQueryPlan) -> AgentResponse: ...
```

Every specialist agent **extends** `BaseAgent`. The pipeline doesn't care
_which_ agent it's calling — it just calls `.run(plan)` on all of them.
This is the **Liskov Substitution Principle** in action.

### 2. `AgentRegistry` — Service Discovery

```python
registry = AgentRegistry()
registry.register(PolicyRAGAgent(...))   # registers + calls .initialize()
registry.register(DataQueryAgent())

agent = registry.get(AgentName.POLICY_RAG)
response = agent.run(plan)
```

The OrchestratorAgent decides **which** agents to call.
The Registry tells the executor **where** they are.
New agents can be added without touching the pipeline code.

### 3. `ParallelAgentExecutor` — Fan-Out / Fan-In

```
Sequential:  PolicyRAG(3s) → DataQuery(1s) → Onboarding(2s) = 6s total
Parallel:    All three at once                               = ~3s total
```

```python
executor = ParallelAgentExecutor(registry, max_workers=4)
responses, tasks = executor.execute(plan, plan.agents_to_invoke)

for task in tasks:
    print(f"{task.agent_name.value}: {task.latency_ms:.0f}ms [{task.status.value}]")
```

### 4. `AgentMessage` — Typed Communication

```python
msg = AgentMessage(
    sender    = AgentName.ORCHESTRATOR,
    recipient = AgentName.POLICY_RAG,
    payload   = plan,
    msg_type  = "request",
)
```

Agents communicate through **typed messages** rather than direct calls.
This enables logging, async queuing, and retry logic.

---

## Component-by-Component Walkthrough

### Component A — HR Knowledge Base (`component_a_hr_indexing.py`)

**What it does:** Loads 6 HR policy markdown files, splits them into chunks,
and builds two indexes for retrieval.

```
HR Policy Docs (.md)
      ↓
  load_hr_documents()    — reads files, infers HR category
      ↓
  chunk_hr_document()    — splits on section boundaries (## ###)
      ↓
  generate_embeddings()  — all-MiniLM-L6-v2 (384-dim vectors)
      ↓
  build_faiss_index()    — FAISS IndexFlatIP for vector search
  build_bm25_index()     — BM25Okapi for keyword search
      ↓
  save_index()           — data/index/ (shared by all agents)
```

**Key concepts:**
- **Chunking** — Splitting long documents into ~300-word pieces so retrieval
  returns precise policy clauses, not entire documents.
- **Bi-encoder embeddings** — Sentence-level semantic vectors (fast, ~90ms).
- **FAISS** — Facebook AI Similarity Search; answers "which 9 vectors are
  closest to this query vector?" in milliseconds.
- **BM25** — Classical keyword search algorithm (TF-IDF + length normalisation).

**Run it:**
```bash
python component_a_hr_indexing.py   # builds data/index/
```

---

### Component B — OrchestratorAgent (`component_b_orchestrator_agent.py`)

**What it does:** Reads the employee's question and decides:
- _What is the intent?_ (leave? compensation? onboarding?)
- _Which agents should handle it?_
- _Can it be split into independent sub-questions?_

```
Employee Question
      ↓
  classify_intent_rules()     — regex keyword matching on 9 intents
  OR orchestrate_with_llm()   — Ollama Mistral 7B structured JSON output
      ↓
  decompose_query_rules()     — split "X and also Y" into [X, Y]
      ↓
  HRQueryPlan {
    intent,          ← QueryIntent enum
    agents_to_invoke,← which BaseAgents to call
    sub_queries,     ← list of atomic sub-questions
    priority_docs,   ← which HR domains to retrieve from
    needs_structured ← does this need CSV/JSON data?
  }
```

**Key concepts:**
- **Intent classification** — mapping free text to a finite set of categories.
- **Query decomposition** — breaking "multi-hop" questions into independent
  sub-questions each agent can answer separately.
- **Routing table** — deterministic mapping from intent → agents:
  ```python
  ROUTING_TABLE = {
      QueryIntent.LEAVE_POLICY:  [PolicyRAGAgent, ComplianceGuardAgent],
      QueryIntent.COMPENSATION:  [PolicyRAGAgent, DataQueryAgent, ComplianceGuardAgent],
      QueryIntent.HEADCOUNT_DATA:[DataQueryAgent],
      ...
  }
  ```
- **LLM-first + rule fallback** — tries Ollama; falls back to regex rules if
  Ollama is not running (100% operational without an LLM).

**The `OrchestratorAgent` class** extends `PlannerAgent` (not `BaseAgent`)
because it _plans_ work for other agents rather than _answering_ questions.

**Run it:**
```bash
python component_b_orchestrator_agent.py
```

---

### Component C — Specialist Retrieval Agents (`component_c_policy_data_agents.py`)

Two agents, both extending `BaseAgent`, run **in parallel** via `ParallelAgentExecutor`.

#### `PolicyRAGAgent` — Hybrid Retrieval

```
Query Text
    ├──→ FAISS vector search  (top-9 semantically similar chunks)
    └──→ BM25 keyword search  (top-7 keyword-matching chunks)
                ↓
       RRF Fusion (Reciprocal Rank Fusion)
                ↓
         Top-6 deduplicated chunks
```

**Why Hybrid (Vector + Keyword)?**

| Vector Search (FAISS) | Keyword Search (BM25) |
|---|---|
| Finds _semantically_ similar chunks | Finds chunks with _exact_ terms |
| `"annual leave entitlement"` matches `"vacation days"` | `"carry-forward"` finds that exact phrase |
| Score range: 0.85 – 0.97 (narrow) | Score range: 0 – 50+ (wide) |
| Misses exact numbers/codes | Finds "8 days", "Band B4" precisely |

**Why RRF (Reciprocal Rank Fusion)?**

You can't add a vector score of 0.91 to a BM25 score of 23 — they're on
completely different scales. RRF uses **ranks** instead of scores:

```
RRF(chunk) = 1/(60 + vector_rank) + 1/(60 + bm25_rank)
```

A chunk ranked #1 by both gets ~0.033; ranked #5 by both gets ~0.031.
Scale-invariant, mathematically sound.

#### `DataQueryAgent` — Structured Data Lookup

Handles questions about numbers that policy documents don't contain:
- `salary_bands.json` — 6 bands (B1–B6) with CTC ranges, notice period, ESOP eligibility
- `headcount.csv` — 9 departments with headcount, attrition, open positions

```python
# Example: "What is the salary range for a Band 4 manager?"
band_match = re.search(r'b(\d)', question, re.IGNORECASE)  # → "4"
results = [b for b in bands if f"B{band_match.group(1)}" == b["band"]]
# Returns: "Band B4 (Manager / Sr Manager): ₹20L – ₹35L CTC · ESOP eligible"
```

**Run it:**
```bash
python component_c_policy_data_agents.py
```

---

### Component D — Compliance Gate (`component_d_compliance_guard.py`)

Two classes with different roles:

#### `OnboardingAgent` — New Joiner Specialist (extends `BaseAgent`)

Scopes retrieval to onboarding docs only and augments with a structured
checklist. New joiners ask procedural questions ("what do I bring on Day 1?")
that a general agent tends to over-retrieve for.

#### `ComplianceGuardAgent` — Three-Stage Safety Pipeline (does NOT extend `BaseAgent`)

This is a **pipeline stage** (guard), not a specialist retriever. Its
interface takes `(question, chunks, agent_response)` — not just a plan.

**Stage 1 — Cross-Encoder Reranking**

```
Retrieved Chunks (from PolicyRAGAgent)
      ↓
  cross-encoder/ms-marco-MiniLM-L6-v2
  Evaluates query + chunk TOGETHER (joint encoding)
      ↓
  Re-sorted chunks by relevance score
```

_Bi-encoder_ (Component A): fast, independent encoding, good recall.
_Cross-encoder_ (here): slow, joint encoding, excellent precision.
Classic two-stage retrieval pipeline.

**Stage 2 — NLI Fact-Check**

```
Each chunk
      ↓
  cross-encoder/nli-deberta-v3-small
  ENTAILMENT / NEUTRAL / CONTRADICTION
      ↓
  Only ENTAILMENT chunks (score ≥ 0.40) pass to Synthesizer
```

_Why?_ BM25 often retrieves chunks that share terms but answer a different
question (e.g., "sick leave" chunk when the question is about "carry-forward").
NLI catches this — the chunk is NEUTRAL, not ENTAILMENT.

**Stage 3 — Legal Compliance Scan**

```python
SENSITIVE_PATTERNS = {
    "posh":       r"\b(posh|sexual harassment|misconduct)\b",
    "termination":r"\b(terminat|fired|let go)\b",
    "pip":        r"\b(pip|performance improvement)\b",
    ...
}
```

- **Caveat injection** for sensitive topics (e.g., POSH → contact ICC)
- **Hard block** if the question shows intent to file a POSH complaint

**Run it:**
```bash
python component_d_compliance_guard.py
```

---

### Component E — Response Synthesis + RAGAS Evaluation (`component_e_response_synthesizer.py`)

**Synthesis:**

```
Agent Responses (from C + D) + Verified Chunks
      ↓
  synthesize_with_ollama()   — Mistral 7B with intent-specific format hints
  OR synthesize_template()   — keyword extraction fallback (no LLM needed)
      ↓
  Append compliance caveats
      ↓
  FinalHRResponse
```

**RAGAS Evaluation (local, no API key):**

| Metric | What it measures | How computed |
|---|---|---|
| **Faithfulness** | Is the answer grounded in context? | NLI entailment of answer sentences vs context |
| **Answer Relevancy** | Does the answer address the question? | Cosine similarity (question embedding, answer embedding) |
| **Context Precision** | Are retrieved chunks all useful? | Fraction of chunks that contribute to the answer |

```
Faithfulness ≥ 0.80 → CI/CD gate PASS
Faithfulness < 0.80 → CI/CD gate FAIL (deployment blocked)
```

**Run it:**
```bash
python component_e_response_synthesizer.py
```

---

## How All 5 Agents Work Together

```
Employee: "What is my leave carry-forward limit and can I also
           check my salary band as a Band 3 employee?"

[B] OrchestratorAgent.plan()
    Intent:  MULTI_DOMAIN
    Agents:  [PolicyRAGAgent, DataQueryAgent, ComplianceGuardAgent]
    Sub-Qs:  ["What is the leave carry-forward limit?",
               "What is the salary band for Band 3?"]

[C] ParallelAgentExecutor.execute()          ← CONCURRENT
    Thread 1: PolicyRAGAgent.run(plan)
      FAISS + BM25 → RRF → 6 chunks from leave_policy.md
    Thread 2: DataQueryAgent.run(plan)
      salary_bands.json → "B3 (Senior Lead): ₹12L–₹20L CTC"
    (Thread 1 and Thread 2 run simultaneously, not sequentially)

[D] ComplianceGuardAgent.run()               ← SEQUENTIAL GATE
    Cross-encoder reranks 6 chunks
    NLI: 4/6 pass ENTAILMENT ≥ 0.40
    Scan: no sensitive topics
    passes=True, flags=[]

[E] ResponseSynthesizerAgent                 ← MERGES ALL OUTPUTS
    Merges PolicyRAG answer + DataQuery table
    Synthesizes with Mistral 7B (or template fallback)
    Evaluates: faithfulness=0.87, relevancy=0.82, precision=0.75

Employee sees:
  "Annual leave carry-forward limit is 8 days (encashable up to 4 days).
   Your Band 3 (Senior Lead) salary range is ₹12L – ₹20L CTC.
   Source: leave_policy.md, salary_bands.json"
```

---

## CLI Commands

```bash
# One-time: build FAISS + BM25 index
python hr_copilot_pipeline.py --build-index

# Interactive mode (type questions)
python hr_copilot_pipeline.py

# Single question
python hr_copilot_pipeline.py --question "What is the maternity leave policy?"

# Run 6-question RAGAS evaluation suite
python hr_copilot_pipeline.py --eval

# Eval + CI/CD gate (exit code 1 if faithfulness < 0.80)
python hr_copilot_pipeline.py --eval --gate
```

---

## Run Each Phase Individually (Workshop Mode)

```bash
python component_a_hr_indexing.py           # Phase 1: FAISS + BM25 index
python component_b_orchestrator_agent.py    # Phase 2: OrchestratorAgent routing
python component_c_policy_data_agents.py    # Phase 3: PolicyRAG + DataQuery
python component_d_compliance_guard.py      # Phase 4: Rerank + NLI + Legal
python component_e_response_synthesizer.py  # Phase 5: Synthesize + RAGAS
python hr_copilot_pipeline.py               # Phase 6: Full pipeline CLI
python agent_framework.py                   # Framework: test BaseAgent + Registry
streamlit run hr_copilot_ui.py              # UI: Streamlit web app
```

---

## The Streamlit Web UI (5 Tabs)

| Tab | What it shows |
|---|---|
| **💬 Chat** | Ask any HR question; see agent badges + RAGAS scores |
| **🎯 Scenarios** | 40 pre-built HR questions across 8 domains |
| **📊 Eval Suite** | 6-question RAGAS benchmark + CI/CD gate result |
| **📚 Knowledge Base** | Browse HR policies, salary bands, headcount data |
| **🏗️ Architecture** | Interactive 5-agent pipeline diagram |

---

## Local vs Azure Production

| Component | Local (this repo) | Azure Production |
|---|---|---|
| Embeddings | all-MiniLM-L6-v2 (384-dim) | text-embedding-3-large (3072-dim) |
| Vector Search | FAISS flat index | Azure AI Search (HNSW) |
| Keyword Search | BM25Okapi | Azure AI Search full-text |
| LLM | Mistral 7B (Ollama) | Azure OpenAI GPT-4o |
| Reranker | ms-marco-MiniLM-L6-v2 | Azure AI Search semantic ranker |
| NLI Model | nli-deberta-v3-small | Azure AI Language classification |
| Agent Parallelism | ThreadPoolExecutor | Azure Durable Functions fan-out |
| Registry | In-process AgentRegistry | Azure API Management |
| Messages | In-process AgentMessage | Azure Service Bus |
| UI | Streamlit (localhost) | Azure Container Apps |

---

## 40 Sample Questions to Try

### Leave Policy
1. What is the maximum annual leave carry-forward and can it be encashed?
2. How many sick leave days am I entitled to per year?
3. What is the maternity leave duration?
4. Can I combine casual leave and annual leave for a long trip?
5. What bereavement leave am I entitled to if a parent passes away?

### Compensation & Benefits
6. What is the salary band range for a Band 3 Senior Lead?
7. What is the CTC range for a Band 4 Manager and are they ESOP eligible?
8. Which monthly allowances are tax-exempt?
9. How is variable pay calculated and when is it paid?
10. What is the performance rating and increment matrix?

### Remote Work
11. Can I work from home 3 days a week as a confirmed employee?
12. Can I work from abroad for a month? What approvals do I need?
13. What are the core working hours I must be available during?
14. I am on probation — how many days must I come to office?
15. How do I apply for a recurring weekly remote work schedule?

### Learning & Development
16. What is my annual L&D budget?
17. How do I claim reimbursement for an AWS certification exam?
18. Does unused L&D budget carry forward to next year?
19. What internal learning programs does Enterprise Corp offer?
20. What is the clawback policy if I leave after a company-sponsored certification?

### POSH — Hard Block (watch the compliance guard)
21. I want to file a POSH complaint against my manager.
22. I need to register a sexual harassment complaint — who do I contact?
23. My colleague made inappropriate advances — how do I file formally?
24. How do I lodge a misconduct complaint about workplace harassment?

### POSH — Caveat Only
25. What is the company's POSH policy?
26. Is POSH training mandatory for all employees?

### Termination — Caveat Trigger
27. What happens if I am terminated during my probation period?
28. Can I be fired for consistently poor performance reviews?
29. What is the notice period if I am dismissed without cause?

### PIP — Caveat Trigger
30. I have been placed on a PIP — what are my rights and next steps?
31. What happens if I fail my performance improvement plan?

### Legal Action — Caveat Trigger
32. Can I sue the company for wrongful termination?
33. How do I escalate a grievance to the labour court?
34. What is the whistleblower protection policy?

### Salary — Caveat Trigger
35. Can my salary be cut if I move to a lower band role?
36. Is the company allowed to reduce my CTC without my consent?

### Out-of-Scope
37. What is the weather forecast for tomorrow?
38. Can you book a flight ticket to Mumbai for me?

### Multi-Domain
39. I am on a PIP and want to know if I can take annual leave and work remotely.
40. If I get terminated, does my ESOP vest and can I encash my remaining leave?

---

## Key Concepts Reference

| Concept | Where used | Why it matters |
|---|---|---|
| **Multi-Agent System** | Entire project | Specialist agents outperform monolithic LLM |
| **BaseAgent / Interface** | agent_framework.py | Uniform interface = easy to add new agents |
| **AgentRegistry** | agent_framework.py | Service discovery without hardcoded references |
| **ParallelAgentExecutor** | hr_copilot_pipeline.py | Fan-out reduces wall-clock latency |
| **Hybrid Retrieval (RAG)** | component_c | Vector + keyword = better recall |
| **RRF Fusion** | component_c | Scale-invariant rank merging |
| **Cross-Encoder Reranking** | component_d | Two-stage retrieve-then-rank |
| **NLI Fact-Check** | component_d | Prevents hallucinated answers |
| **Compliance Gate** | component_d | Legal liability mitigation |
| **RAGAS Evaluation** | component_e | CI/CD quality gate (no API key needed) |

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: faiss` | `pip install faiss-cpu` |
| `ModuleNotFoundError: streamlit` | `pip install streamlit` |
| `ModuleNotFoundError: agent_framework` | Ensure you're running from the `HR_Copilot/` directory |
| Ollama not found | Install from https://ollama.com — template fallback works without it |
| Index not found | Run `python component_a_hr_indexing.py` first |
| Port 8501 busy | `streamlit run hr_copilot_ui.py --server.port 8502` |
| Answer shows "No relevant policy found" | Rebuild index: `python component_a_hr_indexing.py` |
| Agents don't run in parallel | Ensure `agent_framework.py` is in the same directory |

---

*HR Copilot · Multi-Agent Framework Edition · Maneesh Kumar*
