# HR Copilot — Workshop Build Guide
### Multi-Agent Framework Edition
#### Maneesh Kumar

---

## Overview

This guide walks through building a **production-grade multi-agent HR Copilot** from scratch.
Each section maps to one Python file. Students build and test each component independently
before assembling the full pipeline.

**What makes this multi-agent?**
- 5 agents with distinct responsibilities, each independently testable
- A proper framework (`agent_framework.py`) with `BaseAgent`, `AgentRegistry`, and `ParallelAgentExecutor`
- Specialist agents run **concurrently** (fan-out pattern, not sequential)
- Typed contracts (`HRQueryPlan`, `AgentResponse`) prevent hallucination contamination between agents

---

## Problem Statement

### The Challenge

Enterprise HR teams spend 60–70% of their time answering **repetitive, policy-level questions**:

> *"How many sick leave days do I have?"*
> *"Can I work from abroad for two weeks?"*
> *"What happens to my L&D budget if I don't use it?"*

A single monolithic LLM cannot reliably answer all of these because:

| Limitation | Why it matters |
|---|---|
| HR spans 9+ domains | Leave, Compensation, Remote Work, Compliance, L&D, Onboarding — each has distinct rules |
| Policy docs + structured data | Some answers need PDF/Markdown retrieval; others need exact numbers from CSV/JSON |
| Legal and compliance risk | Wrong advice on POSH, termination, or salary creates legal liability |
| Hallucination | LLMs confidently produce wrong numbers (leave days, salary bands, timelines) |
| Multi-domain questions | "Take leave AND check my increment" spans 2 domains simultaneously |

### The Solution: Multi-Agent HR Copilot

```
Employee Question
      │
      ▼
[Framework] agent_framework.py
   BaseAgent · PlannerAgent · AgentRegistry · ParallelAgentExecutor
      │
      ▼
[A] Knowledge Base     ← index HR policy docs + structured data
      │
      ▼
[B] OrchestratorAgent  ← classify intent, decide which agents to invoke
 (PlannerAgent)
      │
      ├──► [C1] PolicyRAGAgent      ← hybrid RAG (FAISS + BM25 + RRF)
      ├──► [C2] DataQueryAgent      ← exact lookup over CSV/JSON data
      └──► [C3] OnboardingAgent     ← new joiner specialist
      (all 3 run CONCURRENTLY via ParallelAgentExecutor)
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

## File Map

| File | Role | Framework class |
|---|---|---|
| `agent_framework.py` | Multi-agent backbone | `BaseAgent`, `AgentRegistry`, `ParallelAgentExecutor` |
| `hr_data_models.py` | Shared typed contracts | Dataclasses + Enums |
| `component_a_hr_indexing.py` | Knowledge Base | — |
| `component_b_orchestrator_agent.py` | Intent routing | `OrchestratorAgent(PlannerAgent)` |
| `component_c_policy_data_agents.py` | Retrieval agents | `PolicyRAGAgent(BaseAgent)`, `DataQueryAgent(BaseAgent)` |
| `component_d_compliance_guard.py` | Guard + Onboarding | `OnboardingAgent(BaseAgent)`, `ComplianceGuardAgent` |
| `component_e_response_synthesizer.py` | Synthesis + Eval | standalone functions |
| `hr_copilot_pipeline.py` | Pipeline entry point | Uses registry + executor |
| `hr_copilot_ui.py` | Streamlit UI | — |

---

## Phase 0 — Multi-Agent Framework (`agent_framework.py`)

> **Build this first. All agents depend on it.**

### What to build

The framework is HR-agnostic — it contains only the architectural skeleton.

**1. `BaseAgent` (Abstract Base Class)**

```python
class BaseAgent(ABC):
    @property
    @abstractmethod
    def agent_name(self) -> AgentName: ...

    @abstractmethod
    def run(self, plan: HRQueryPlan) -> AgentResponse: ...

    def initialize(self) -> None: pass   # optional setup
    def shutdown(self) -> None:   pass   # optional cleanup
```

_Design principle:_ Every specialist agent implements this same interface.
The pipeline calls `agent.run(plan)` without knowing which agent it's calling.
This is the **Liskov Substitution Principle** — all BaseAgent subclasses are interchangeable from the caller's perspective.

**2. `PlannerAgent` (Routing Agent Base)**

```python
class PlannerAgent(ABC):
    @abstractmethod
    def plan(self, question: str) -> HRQueryPlan: ...
```

Separate from `BaseAgent` because its output is a plan, not an answer.

**3. `AgentRegistry` (Service Discovery)**

```python
registry = AgentRegistry()
registry.register(PolicyRAGAgent(...))   # calls agent.initialize()
registry.register(DataQueryAgent())
agent = registry.get(AgentName.POLICY_RAG)
```

The registry is the single source of truth for all agents in the system.
New agents can be added without changing the pipeline.

**4. `ParallelAgentExecutor` (Fan-Out/Fan-In)**

```python
executor = ParallelAgentExecutor(registry, max_workers=4)
responses, tasks = executor.execute(plan, plan.agents_to_invoke)
```

Uses `ThreadPoolExecutor` to submit all specialist agents simultaneously.
Results collected with `as_completed()` as each finishes.

```
Sequential:  RAG(3s) → Data(1s) → Onboarding(2s) = 6s total
Parallel:    all three at once                    = ~3s total
```

**5. `AgentMessage` (Inter-Agent Communication)**

```python
msg = AgentMessage(
    sender    = AgentName.ORCHESTRATOR,
    recipient = AgentName.POLICY_RAG,
    payload   = plan,
    msg_type  = "request",
)
```

Typed messages provide: loose coupling, auditability, async-ready design.

### Challenge exercises

**Challenge F0-1 — Add a new BaseAgent**
Implement a `SalaryCalculatorAgent(BaseAgent)` that:
- Accepts a plan with `intent = COMPENSATION`
- Reads `salary_bands.json` and calculates take-home from gross CTC
- Returns an `AgentResponse` with the calculation
- Register it in the registry and verify it appears in `executor.execute()`

**Challenge F0-2 — Registry health check**
Add a `health_check()` method to `AgentRegistry` that:
- Calls `agent.run()` with a minimal test plan
- Returns a dict `{agent_name: "healthy" | "error: <msg>"}`
- Use it in the pipeline to skip failed agents gracefully

**Edge case: What happens when an agent raises an exception mid-execution?**
```python
# In your test, make PolicyRAGAgent.run() raise ValueError
# Verify: ParallelAgentExecutor catches it, marks task as FAILED,
# but still returns results from DataQueryAgent and OnboardingAgent
```

---

## Component A — HR Knowledge Base (`component_a_hr_indexing.py`)

**Goal:** Convert HR policy documents into a searchable FAISS + BM25 index
that all agents can query.

### What to build

1. **Document loader** — reads `.md`, `.pdf`, `.docx` files from `data/hr_docs/`
2. **HR-aware chunker** — splits at section boundaries (`\n## `, `\n### `)
   - Chunk size: ~300 words → keeps individual policy clauses intact
   - Overlap: ~100 words → prevents clause split across boundaries
3. **Embedding model** — `all-MiniLM-L6-v2` (384-dim, local, no API key)
4. **FAISS flat index** — `IndexFlatIP` with L2-normalised vectors
5. **BM25 keyword index** — `BM25Okapi` with HR-specific tokenisation
6. **Save to disk** — `data/index/hr_faiss.index`, `hr_bm25.pkl`, `hr_chunks.json`

### HR category mapping

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

Each chunk is tagged with its category. This tag is used by agents to
filter retrieval to only the relevant HR domain.

### Expected output

```
Total HR documents: 6
Total chunks: 109
  [compensation]:   18 chunks
  [grievance    ]:  19 chunks
  [learning     ]:  20 chunks
  [leave        ]:  18 chunks
  [onboarding   ]:  17 chunks
  [remote_work  ]:  17 chunks
```

### Challenge exercises

**Challenge A-1 — Add a new policy document**
Create `data/hr_docs/travel_policy.md` with at least 3 sections covering
domestic travel allowances, international travel approvals, and expense
reimbursement. Run Component A and verify it's indexed in a new `travel`
category. Then ask the pipeline: `"What is the business travel allowance?"`.

**Challenge A-2 — Chunk size ablation**
Change `CHUNK_SIZE` to 150 words and 600 words. Run the eval suite both times:
```bash
python hr_copilot_pipeline.py --eval
```
Measure how faithfulness and context_precision change. Which chunk size
gives the best trade-off and why?

**Edge / failure cases to test:**

| Scenario | How to trigger | Expected behaviour |
|---|---|---|
| Empty document | Create `data/hr_docs/empty.md` with no content | Loader skips it, no chunks added |
| Duplicate chunks | Run `component_a_hr_indexing.py` twice | Second run overwrites index; no duplicates |
| Missing index at query time | Delete `data/index/hr_faiss.index` then `ask()` | Pipeline rebuilds index automatically |
| Very short document (<100 words) | Create a tiny `.md` | One chunk created, category inferred correctly |
| Binary/non-text file | Add a `.png` to `hr_docs/` | Loader raises warning and skips, does not crash |

---

## Component B — OrchestratorAgent (`component_b_orchestrator_agent.py`)

**Goal:** Understand the employee's intent and route the query to the right
specialist agents. Produces a typed `HRQueryPlan` consumed by all downstream
agents.

### What to build

1. **9 HR intent classes** via `QueryIntent` enum
2. **Rule-based classifier** — regex keyword matching per intent
3. **Query decomposer** — splits "X and also Y" into independent sub-questions
4. **Routing table** — deterministic mapping intent → agents
5. **`OrchestratorAgent(PlannerAgent)`** class — wraps routing logic

### HRQueryPlan — the typed contract

```python
@dataclass
class HRQueryPlan:
    original_question:  str
    intent:             QueryIntent         # LEAVE_POLICY, COMPENSATION, etc.
    sub_queries:        List[str]           # decomposed sub-questions
    agents_to_invoke:   List[AgentName]     # which BaseAgents to call
    priority_docs:      List[str]           # category filter for retrieval
    needs_structured:   bool               # True → invoke DataQueryAgent
    reasoning:          str
    confidence:         float
```

Never pass raw strings between agents. The `HRQueryPlan` is the contract.

### Routing table

```python
ROUTING_TABLE = {
    QueryIntent.LEAVE_POLICY:   [PolicyRAGAgent, ComplianceGuardAgent],
    QueryIntent.COMPENSATION:   [PolicyRAGAgent, DataQueryAgent, ComplianceGuardAgent],
    QueryIntent.HEADCOUNT_DATA: [DataQueryAgent],
    QueryIntent.SALARY_BAND:    [DataQueryAgent, ComplianceGuardAgent],
    QueryIntent.MULTI_DOMAIN:   [All agents],
    ...
}
```

### Challenge exercises

**Challenge B-1 — Add a new intent**
Add `BENEFITS_QUERY` intent for questions about health insurance, PF, and gratuity.
Steps:
1. Add `BENEFITS_QUERY = "benefits_query"` to `QueryIntent`
2. Add regex pattern to `INTENT_KEYWORDS`
3. Add entry to `ROUTING_TABLE` and `PRIORITY_DOCS_MAP`
4. Test: `"What is the health insurance coverage for my family?"`

**Challenge B-2 — Improve multi-domain decomposition**
The current decomposer splits on `" and also "`. Extend it to handle:
- `"both [X] and [Y]"` → `["What is X?", "What is Y?"]`
- `"[X] versus [Y]"` → `["What is X?", "What is Y?"]`
Test with: `"Compare Band 3 and Band 4 compensation and also leave entitlements"`.

**Edge / failure cases to test:**

| Scenario | Input | Expected behaviour |
|---|---|---|
| Completely unknown question | `"What is the capital of France?"` | `UNKNOWN` intent, `PolicyRAGAgent` fallback |
| Single-word question | `"leave"` | `LEAVE_POLICY` intent, no decomposition |
| Question with 5+ matched intents | Long question with leave + comp + remote + L&D + onboarding terms | `MULTI_DOMAIN` intent |
| Ollama unavailable | Kill Ollama process, call with `use_llm=True` | Graceful fallback to rule-based classifier |
| Empty question string | `""` | `UNKNOWN` intent, no crash |
| POSH complaint question | `"I want to file a POSH complaint"` | `GRIEVANCE` intent, `ComplianceGuardAgent` included |

---

## Component C — Specialist Agents (`component_c_policy_data_agents.py`)

**Goal:** Two `BaseAgent` subclasses run in parallel, each retrieving from
its own data source.

### PolicyRAGAgent — Hybrid Retrieval

Both agents extend `BaseAgent`, so `ParallelAgentExecutor` can call `.run(plan)` on them concurrently.

```
Query Text
    ├──→ FAISS vector search  (top-9 semantically similar chunks)
    └──→ BM25 keyword search  (top-7 keyword-matching chunks)
               ↓
      RRF Fusion (Reciprocal Rank Fusion)
      RRF(chunk) = 1/(60 + vector_rank) + 1/(60 + bm25_rank)
               ↓
        Top-6 deduplicated chunks → AgentResponse
```

**Why RRF over score normalisation?**
FAISS cosine scores cluster at 0.85–0.97. BM25 scores range 0–50+.
Adding them directly gives wrong weights. RRF uses **rank order only** —
completely scale-invariant.

**Two-pass category filter:**
Pass 1: Return chunks from priority category.
Pass 2: Backfill with chunks from other categories if pass 1 insufficient.
This gives MULTI_DOMAIN questions per-sub-query scoping.

### DataQueryAgent — Exact Lookup

Answers questions needing precise numbers that policy docs don't contain:

```python
# "What is the CTC for Band 4 manager?"
band_match = re.search(r'b(\d)', question, re.IGNORECASE)
band = f"B{band_match.group(1)}"   # → "B4"
results = [b for b in bands if b["band"] == band]
# Returns: "Band B4: ₹20L – ₹35L CTC · ESOP eligible"
```

### Challenge exercises

**Challenge C-1 — Implement ColBERT-style late interaction**
The current RRF uses simple rank fusion. Implement a scoring alternative
where you compute the MaxSim between each query token embedding and chunk
token embeddings, and use this as a third signal alongside FAISS + BM25.
Test whether average faithfulness improves.

**Challenge C-2 — Add NL-to-SQL for DataQueryAgent**
Currently DataQueryAgent uses regex to detect "Band 4" patterns.
Replace the routing logic with a pandas `.query()` call, e.g.:
```python
df.query("band == 'B4' and esop_eligible == True")
```
Handle these questions:
- `"Which bands are ESOP eligible?"` → `df[df.esop_eligible == True]`
- `"Departments with attrition > 10%"` → filter headcount.csv

**Challenge C-3 — Caching layer**
PolicyRAGAgent re-encodes the query on every call. Add an LRU cache:
```python
from functools import lru_cache
@lru_cache(maxsize=128)
def _encode_query(self, query: str) -> np.ndarray: ...
```
Measure latency improvement for repeated identical queries.

**Edge / failure cases to test:**

| Scenario | Input | Expected behaviour |
|---|---|---|
| Zero chunks retrieved | Delete `data/index/hr_chunks.json` | Returns `AgentResponse` with "No relevant policy found", no crash |
| BM25 returns all-zero scores | Gibberish query with no HR keywords | Falls back to FAISS-only results via RRF |
| Salary band not found | `"What is Band B9 salary?"` | Returns full band table (all bands), no crash |
| Headcount CSV missing | Delete `headcount.csv`, ask about headcount | Returns "Headcount data not available" message |
| Concurrent access from two threads | Run two `.run(plan)` calls simultaneously | Thread-safe — independent FAISS calls, no shared state |

---

## Component D — Compliance Gate (`component_d_compliance_guard.py`)

**Goal:** Three-stage safety pipeline — rerank, fact-check, legal scan.
Also contains `OnboardingAgent(BaseAgent)` for new joiner queries.

### OnboardingAgent (extends BaseAgent)

Scopes retrieval to onboarding docs only and augments with a hard-coded
checklist. Registered in `AgentRegistry`, runs in parallel with other agents.

### ComplianceGuardAgent (NOT a BaseAgent)

This is a **pipeline guard stage**, not a specialist retriever.
Its interface: `run(question, chunks, agent_response) → (verified_chunks, ComplianceCheckResult)`

Three stages:

**Stage 1 — Cross-Encoder Reranking**
```
Input: query + 6 retrieved chunks
Model: cross-encoder/ms-marco-MiniLM-L6-v2
Evaluates (query, chunk) JOINTLY — 10× better relevance than bi-encoder
Output: top-8 chunks sorted by joint relevance score
```

**Stage 2 — NLI Fact Check**
```
For each chunk:
  NLI input: f"{chunk.text[:400]} [SEP] {question}"
  Labels:    ENTAILMENT / NEUTRAL / CONTRADICTION
  Threshold: entailment score ≥ 0.40
Output: only ENTAILMENT chunks pass
```

**Stage 3 — Legal Compliance Scan**
```
Scan question + answer for: posh, termination, pip, legal, salary_cut
→ Add mandatory caveats per topic
→ POSH + "file complaint" → hard block → redirect to ICC
```

### Challenge exercises

**Challenge D-1 — Custom compliance rule**
Add a new sensitive topic: `"probation"`. Any question about probation
extension, failure, or termination during probation should add:
`"⚠️ Probation-related decisions require HRBP and manager sign-off."`
Test with: `"Can my probation be extended indefinitely?"`.

**Challenge D-2 — Threshold sensitivity analysis**
Change `NLI_THRESH` from 0.40 to 0.20, 0.40, 0.60, and 0.80.
Run the eval suite at each threshold. Plot faithfulness vs. context_precision.
At what threshold do you get the best trade-off?

**Challenge D-3 — Contradiction detection**
The NLI model also produces CONTRADICTION labels. Add a new feature:
if any chunk has contradiction score > 0.70, log a warning:
`"⚠️ Retrieved context contradicts the question — possible policy update needed"`
Test with deliberately contradictory chunks.

**Edge / failure cases to test:**

| Scenario | Input | Expected behaviour |
|---|---|---|
| POSH policy question (not complaint) | `"What is the POSH policy?"` | Answers with POSH caveat, NOT blocked |
| POSH complaint question | `"I want to file a POSH complaint"` | Hard block, returns ICC contact, `passes=False` |
| Empty chunks list | Pass `[]` to `compliance_guard.run()` | Returns empty verified list, `passes=True`, no crash |
| NLI model fails to load | Simulate import error | Graceful fallback: mark all chunks as verified |
| All chunks fail NLI threshold | Gibberish chunks with no policy content | Returns top-3 chunks as fallback (not empty) |
| Termination + POSH in same question | `"I was harassed and then fired — what are my options?"` | Both flags set, both caveats appended |

---

## Component E — Response Synthesis + RAGAS Evaluation (`component_e_response_synthesizer.py`)

**Goal:** Merge all agent outputs into one clean, employee-facing answer
and evaluate quality with RAGAS-style metrics.

### Synthesis (two paths)

**LLM path (Ollama/Mistral 7B):**
```python
system = SYNTHESIS_SYSTEM.format(format_hint=FORMAT_HINTS[intent])
answer = ollama.chat(model="mistral", messages=[system, user])
```

**Template fallback (no LLM):**
```python
# Phase 1: extract sections whose headers contain question keywords
# Phase 2: if no match, return top-2 chunks with headers → bold
body = _extract_relevant_lines(question, verified_chunks)
```

### RAGAS metrics (local, no API key)

| Metric | Formula | Gate |
|---|---|---|
| Faithfulness | `NLI_entailment(answer_sentences, context)` | ≥ 0.80 |
| Answer Relevancy | `cosine(embed(question), embed(answer))` | warn < 0.68 |
| Context Precision | `fraction of chunks used in answer` | — |

### Challenge exercises

**Challenge E-1 — Intent-specific answer formatter**
Add a new `FORMAT_HINTS` entry for the new `BENEFITS_QUERY` intent you
created in Challenge B-1. Use a table format to present health insurance,
PF, and gratuity side by side.

**Challenge E-2 — Add response length control**
Employees want concise answers (< 200 words). Add a `max_words` parameter
to `synthesize_template()`. Truncate at the last complete sentence within
the limit. Measure whether faithfulness changes with shorter answers.

**Challenge E-3 — Ground-truth evaluation**
Create a `data/eval/ground_truth.json` with 5 questions and verified answers.
Modify the eval suite to calculate **ROUGE-L** and **exact match** scores
against ground truth, in addition to RAGAS metrics.

**Edge / failure cases to test:**

| Scenario | Input | Expected behaviour |
|---|---|---|
| All agents return empty answers | Mock agents returning `answer=""` | Returns "I don't have enough information" message |
| POSH block from compliance | POSH complaint question | Synthesizer returns corrected_answer directly, skips LLM |
| Ollama unavailable | Kill Ollama, `USE_OLLAMA=True` | Falls back to template synthesizer, no crash |
| Answer longer than 2000 chars | Very broad question | Still evaluates RAGAS correctly; faithfulness may drop |
| Zero verified chunks | No chunks pass NLI | Faithfulness = 1.0 (no sentences to check), relevancy normal |
| Faithfulness gate fails | Deliberately bad retrieval | `run_eval_suite(gate=True)` exits with code 1 |

---

## The Multi-Agent Framework in Action

The pipeline wires all components together using the framework:

```python
# From hr_copilot_pipeline.py

# 1. Register all specialist agents
registry = AgentRegistry()
registry.register(PolicyRAGAgent(chunks, faiss, bm25, embed))  # BaseAgent
registry.register(DataQueryAgent())                              # BaseAgent
registry.register(OnboardingAgent(policy_agent))                # BaseAgent

# 2. Create parallel executor
executor = ParallelAgentExecutor(registry, max_workers=4)

# 3. Orchestrator plans the work
orchestrator = OrchestratorAgent()                              # PlannerAgent
plan = orchestrator.plan(question, use_llm=False)

# 4. FAN-OUT: run specialist agents concurrently
responses, tasks = executor.execute(plan, plan.agents_to_invoke)

# 5. Per-agent timing trace
executor.print_execution_summary(tasks)
# ┌─ Parallel Execution Summary ─────────────────────────────────────┐
# │  ✅ PolicyRAGAgent       ████████          1753ms               │
# │  ✅ DataQueryAgent       ██████████        2094ms               │
# │  ✅ OnboardingAgent      ████████          1751ms               │
# │  Total wall-clock time:                    2094ms               │
# └──────────────────────────────────────────────────────────────────┘

# 6. Sequential guard + synthesis
verified, comp = compliance_guard.run(question, chunks)
final = synthesizer_agent(question, plan, responses, verified, comp)
```

### Full system challenge

**Challenge M-1 — Swap in a new retrieval agent without touching the pipeline**

1. Create `component_f_semantic_cache_agent.py`
2. Implement `SemanticCacheAgent(BaseAgent)`:
   - Stores recent (question, answer) pairs as embeddings
   - On each new question, find the most similar cached question
   - If similarity > 0.92, return the cached answer directly
   - Otherwise run the normal pipeline and cache the result
3. Register it in `AgentRegistry` with `AgentName.CACHE` (add to enum)
4. The pipeline should not need any changes — the executor calls `.run(plan)` on it automatically

---

## Sample Questions by Domain

Use these to test each component as you build it, and to exercise the
full pipeline end-to-end. Each group is designed to trigger specific
system behaviour — marked where relevant.

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
14. I am on probation — how many days must I come to the office?
15. How do I apply for a recurring weekly remote work schedule?

### Learning & Development
16. What is my annual L&D budget?
17. How do I claim reimbursement for an AWS certification exam?
18. Does unused L&D budget carry forward to next year?
19. What internal learning programs does Enterprise Corp offer?
20. What is the clawback policy if I leave after a company-sponsored certification?

### POSH — Hard Block *(triggers `ComplianceGuardAgent` — `passes=False`)*
21. I want to file a POSH complaint against my manager.
22. I need to register a sexual harassment complaint — who do I contact?
23. My colleague made inappropriate advances — how do I file formally?
24. How do I lodge a misconduct complaint about workplace harassment?

### POSH — Caveat Only *(informational — passes with ICC caveat)*
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

### Salary Cut — Caveat Trigger
35. Can my salary be cut if I move to a lower band role?
36. Is the company allowed to reduce my CTC without my consent?

### Out-of-Scope *(tests `UNKNOWN` intent fallback — no HR agents invoked)*
37. What is the weather forecast for tomorrow?
38. Can you book a flight ticket to Mumbai for me?
39. Tell me a joke about software engineers.
40. What is the GDP of India in 2024?

### Multi-Domain *(triggers parallel execution across 2–3 agents)*
41. I am on a PIP and want to know if I can take annual leave and work remotely.
42. If I get terminated, does my ESOP vest and can I encash my remaining leave?

---

## Advanced / Demo Showcase Questions

These complex, multi-constraint questions are ideal for live demos.
Each one intentionally crosses 2–4 policy domains, triggers parallel
specialist agents, and (in some cases) the compliance guard.

> **Run these through the Chat tab in the Streamlit UI for the best visual experience.**
> Each shows the agent fan-out, RAGAS metrics, and compliance caveats side-by-side.

### Band & Seniority Deep-Dives

**D1.** As a Band 3 Senior Lead on probation, how many WFH days am I entitled to, what is my L&D budget, and what happens to my sponsored certification clawback if I resign before completing probation?
> *Triggers: remote_work + learning + probation caveat — 3-agent fan-out*

**D2.** Compare Band 3 and Band 4 across all dimensions: salary ranges, ESOP eligibility, notice period, WFH entitlement, and annual L&D budget.
> *Triggers: salary_band + compensation + remote_work — DataQueryAgent + PolicyRAGAgent*

**D3.** I am a Band 2 employee with 18 months of service. If I move to a Band 3 role internally, does my leave balance reset, does my vesting schedule change, and what is my new notice period?
> *Triggers: compensation + leave_policy + salary_band — multi_domain*

### New Joiner Scenarios

**D4.** I joined 3 months ago as a Band 2 Associate — what is my current leave entitlement, how many days must I come to office each week, and what mandatory compliance trainings should I have completed by now?
> *Triggers: leave_policy + remote_work + onboarding — 3-agent fan-out*

**D5.** What is my complete onboarding checklist for the first week, and what is the 30-60-90 day performance plan I should follow?
> *Triggers: onboarding (OnboardingAgent + checklist)*

**D6.** I am joining as a Band 4 Manager next month — what are my exact Day 1 tasks, and what benefits (health insurance, ESOP, L&D) do I need to enroll in within the first 30 days?
> *Triggers: onboarding + compensation — OnboardingAgent + DataQueryAgent*

### Grievance & Compliance Edge Cases

**D7.** My manager has placed me on a PIP the same week I raised an informal grievance against them — what are my rights, what is the escalation path, and who do I contact outside my reporting chain?
> *Triggers: grievance + pip caveat — compliance scan fires, HRBP caveat added*

**D8.** I was terminated during my probation period — am I entitled to any notice pay, can I encash my unused annual leave, and does my ESOP vest at all?
> *Triggers: leave_policy + compensation + grievance — termination caveat + multi_domain*

**D9.** I want to file a formal complaint — my manager has been making comments about my salary that I find inappropriate. Is this a POSH matter or a standard grievance?
> *Triggers: grievance intent — compliance scan; NOTE: if phrased as "file a POSH complaint" the ICC hard-block fires*

**D10.** Can I take sabbatical leave for 3 months to pursue a personal certification, and will the company fund it or count it under my L&D budget?
> *Triggers: leave_policy + learning — multi_domain*

### Multi-Constraint Power Queries

**D11.** I am on a PIP, have 12 days of accumulated annual leave, and am considering resigning. If I resign now, what is my notice period, can I encash all 12 leave days, and what happens to my unvested ESOP?
> *Triggers: leave_policy + compensation + grievance — termination + pip caveats both fire*

**D12.** My team has 3 members: one on maternity leave, one working from abroad, and one on a PIP. What policies govern each of them, and can all three be considered for the annual increment cycle?
> *Triggers: leave_policy + remote_work + compensation + grievance — 4 domains*

**D13.** As a Band 4 Manager, can I approve my own WFH extension, or does it need to go to my skip-level? And what is the maximum duration I can work from abroad in a calendar year?
> *Triggers: remote_work — approval chain + WFA policy*

**D14.** I completed an AWS Solutions Architect certification using my L&D budget. I want to resign 4 months later. Will the company claw back the full course fee, and how is the clawback calculated?
> *Triggers: learning — clawback policy clause*

**D15.** What is the headcount breakdown by department, which department has the highest attrition, and are there salary band differences between Engineering and Sales at Band 3?
> *Triggers: headcount_data + salary_band — DataQueryAgent (structured data)*

---

**How to use these in testing:**

```bash
# Test a single question through the full pipeline
python hr_copilot_pipeline.py --question "What is the salary band for a Band 4 manager?"

# Test the POSH hard block
python hr_copilot_pipeline.py --question "I want to file a POSH complaint against my manager."

# Test multi-domain parallel execution (watch the agent trace)
python hr_copilot_pipeline.py --question "I am on a PIP and want to take annual leave and work remotely."

# Run the advanced demo showcase questions
python hr_copilot_pipeline.py --question "Compare Band 3 and Band 4: salary ranges, ESOP eligibility, notice period, WFH entitlement, and L&D budget."
```

---

## Build Order

```
agent_framework.py      ← Framework backbone (no HR logic)
hr_data_models.py       ← Shared typed contracts
      │
component_a             ← Build the index first
      │
component_b             ← Orchestrator (depends on hr_data_models)
      │
component_c             ← Specialist agents (depend on A + B)
      │
component_d             ← Compliance guard (depends on C)
      │
component_e             ← Synthesizer (depends on all above)
      │
hr_copilot_pipeline.py  ← Wires everything + uses AgentRegistry
      │
hr_copilot_ui.py        ← Streamlit UI (depends on pipeline)
```

### Run order

```bash
# One-time setup
chmod +x setup.sh && ./setup.sh

# Test framework
python agent_framework.py

# Test each component individually
python component_a_hr_indexing.py           # ✅ 109 chunks indexed
python component_b_orchestrator_agent.py    # ✅ 6 routing scenarios pass
python component_c_policy_data_agents.py    # ✅ chunks retrieved per query
python component_d_compliance_guard.py      # ✅ compliance flags + verified chunks
python component_e_response_synthesizer.py  # ✅ full answers + RAGAS scores

# Run full pipeline
python hr_copilot_pipeline.py --eval        # 6-question RAGAS eval suite

# Launch UI
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
| `learning_development.md` | L&D | Rs 25,000 annual budget, eligible certifications, claim process, clawback policy |

---

## Structured Data

| File | Contents | Agent |
|---|---|---|
| `data/hr_structured/salary_bands.json` | CTC min/max, ESOP eligibility, notice period by band B1–B6 | `DataQueryAgent` |
| `data/hr_structured/headcount.csv` | Headcount by department, attrition, open positions, avg tenure | `DataQueryAgent` |

---

## Local vs Azure Production

| Component | Local (Workshop) | Azure Production |
|---|---|---|
| Documents | `.md` files | Azure Blob Storage / SharePoint |
| Embeddings | `all-MiniLM-L6-v2` (384-dim) | `text-embedding-3-large` (3072-dim) |
| Vector search | FAISS `IndexFlatIP` | Azure AI Search (HNSW, built-in hybrid) |
| Keyword search | BM25Okapi | Azure AI Search (built-in BM25) |
| LLM | Mistral 7B via Ollama | Azure OpenAI GPT-4o |
| Reranker | `ms-marco-MiniLM-L6-v2` | Azure AI Search Semantic Ranker (neural) |
| NLI check | `nli-deberta-v3-small` | Azure AI Language (text classification) |
| Agent routing | Rule-based + Ollama | Azure OpenAI structured output (`response_format=JSON`) |
| Agent registry | In-process `AgentRegistry` | Azure API Management |
| Message passing | `AgentMessage` (in-process) | Azure Service Bus |
| Parallel execution | `ThreadPoolExecutor` | Azure Durable Functions fan-out |
| UI | Streamlit (localhost:8501) | Azure Container Apps |
| Evaluation | Local RAGAS | Azure AI Evaluation SDK |

---

## Common Pitfalls

| Pitfall | Cause | Fix |
|---|---|---|
| "No relevant policy found" | Index not built | Run `python component_a_hr_indexing.py` |
| Wrong intent classified | Keyword not in regex | Add to `INTENT_KEYWORDS` in Component B |
| Category filter returns 0 chunks | `priority_docs` values don't match `category` field | Must match exactly: `"leave"`, `"compensation"`, etc. |
| All chunks filtered after reranking | Threshold too high | Set `RERANK_THRESH = 0.0` (keeps all) |
| Markdown headers in UI | Raw chunk text contains `## Section` | Convert to `**bold**` in synthesizer |
| `TypeError: run() takes 2 positional arguments` | Agent doesn't extend `BaseAgent` correctly | Add `agent_name` property to the subclass |
| `AgentRegistry: agent not found` | Agent not registered before `executor.execute()` | Call `registry.register(agent)` in `_ensure_loaded()` |
| Parallel execution returns empty list | All agents have `AgentStatus.FAILED` | Check each agent's `task.error` field for the root cause |
| Segfault loading 3 ML models together | macOS semaphore + multiprocessing conflict | `TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1` |

---

## Consolidated Challenge List

| ID | Component | Difficulty | What you build |
|---|---|---|---|
| F0-1 | Framework | ⭐⭐ | New `SalaryCalculatorAgent(BaseAgent)` |
| F0-2 | Framework | ⭐⭐⭐ | `health_check()` for `AgentRegistry` |
| A-1 | Knowledge Base | ⭐ | Add travel policy document |
| A-2 | Knowledge Base | ⭐⭐ | Chunk size ablation study |
| B-1 | Orchestrator | ⭐⭐ | New `BENEFITS_QUERY` intent |
| B-2 | Orchestrator | ⭐⭐ | Improve multi-domain decomposer |
| C-1 | Retrieval | ⭐⭐⭐ | ColBERT-style late interaction scoring |
| C-2 | Retrieval | ⭐⭐ | NL-to-SQL for DataQueryAgent |
| C-3 | Retrieval | ⭐ | LRU cache for query embeddings |
| D-1 | Compliance | ⭐ | New `probation` sensitive topic |
| D-2 | Compliance | ⭐⭐ | NLI threshold sensitivity analysis |
| D-3 | Compliance | ⭐⭐⭐ | Contradiction detection + warning |
| E-1 | Synthesizer | ⭐ | Format hint for new intent |
| E-2 | Synthesizer | ⭐⭐ | Response length control |
| E-3 | Synthesizer | ⭐⭐⭐ | Ground-truth ROUGE-L evaluation |
| M-1 | Multi-agent | ⭐⭐⭐ | `SemanticCacheAgent` swap-in |

---

*HR Copilot · Multi-Agent Framework Edition · Maneesh Kumar*
