# HR Copilot — Multi-Agent Employee Self-Service System

---

## Quick Start — 3 Commands

### Mac / Linux
```bash
chmod +x setup.sh && ./setup.sh    # one-time setup
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

## The Streamlit Web UI (5 Tabs)

| Tab | What it does |
|---|---|
| **💬 Chat** | Type any HR question → answer with agent badges + RAGAS scores |
| **🎯 Scenarios** | 40 pre-built HR questions across 8 domains — click any to run |
| **📊 Eval Suite** | 6-question RAGAS benchmark with CI/CD gate (faithfulness ≥ 0.80) |
| **📚 Knowledge Base** | Browse HR policies, salary band table, headcount chart |
| **🏗️ Architecture** | Interactive 5-agent pipeline explainer |

---

## Run Each Phase Individually (Workshop Mode)

```bash
python component_a_hr_indexing.py           # Phase 1: FAISS + BM25 index
python component_b_orchestrator_agent.py    # Phase 2: OrchestratorAgent routing
python component_c_policy_data_agents.py    # Phase 3: PolicyRAG + DataQuery + Onboarding
python component_d_compliance_guard.py      # Phase 4: Rerank + NLI + POSH guard
python component_e_response_synthesizer.py  # Phase 5: Synthesize + RAGAS evaluate
python hr_copilot_pipeline.py               # Phase 6: CLI pipeline
streamlit run hr_copilot_ui.py              # Phase 6: Streamlit UI
```

---

## CLI Commands

```bash
python hr_copilot_pipeline.py --build-index        # rebuild index
python hr_copilot_pipeline.py                       # interactive mode
python hr_copilot_pipeline.py --question "..."      # single question
python hr_copilot_pipeline.py --eval                # eval suite
python hr_copilot_pipeline.py --eval --gate         # eval + CI/CD gate
```

---

## What setup.sh Does

1. Creates Python virtual environment (.venv)
2. Installs all dependencies (requirements.txt)
3. Downloads 3 AI models (~350 MB cached)
4. Creates 6 HR policy docs + structured data
5. Builds FAISS + BM25 knowledge index
6. Optionally installs Ollama + Mistral 7B

---

## Folder Structure

```
hr_copilot/
├── hr_data_models.py                    shared typed dataclasses
├── component_a_hr_indexing.py           Phase 1: index
├── component_b_orchestrator_agent.py    Phase 2: orchestrate
├── component_c_policy_data_agents.py    Phase 3: retrieve
├── component_d_compliance_guard.py      Phase 4: guard
├── component_e_response_synthesizer.py  Phase 5: synthesize + eval
├── hr_copilot_pipeline.py               Phase 6: CLI
├── hr_copilot_ui.py                     Phase 6: Streamlit UI
├── requirements.txt
├── setup.sh / setup.bat
└── data/
    ├── hr_docs/          6 HR policy .md files
    ├── hr_structured/    salary_bands.json + headcount.csv
    ├── index/            FAISS + BM25 (built by Phase 1)
    └── eval/             RAGAS reports
```

---

## Local vs Azure

| Component | Local | Azure Production |
|---|---|---|
| Embeddings | all-MiniLM-L6-v2 | text-embedding-3-large |
| Vector Search | FAISS | Azure AI Search (HNSW) |
| LLM | Mistral 7B (Ollama) | Azure OpenAI GPT-4o |
| Evaluation | Local RAGAS | Azure AI Evaluation SDK |
| UI | Streamlit (localhost) | Azure Container Apps |

---

## 20 Sample Questions to Try (Dry Run)

Paste any of these into the **💬 Chat** tab or click the matching scenario in **🎯 Scenarios**:

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

## 20 Negative & Edge-Case Questions (Compliance Guard Testing)

These queries test the compliance guard, POSH blocker, caveats, boundary routing, and failure modes. Paste them into the **💬 Chat** tab to observe how the system handles sensitive, out-of-scope, or adversarial inputs.

### POSH — Hard Block (should be blocked entirely)
21. I want to file a POSH complaint against my manager.
22. I need to register a sexual harassment complaint — who do I contact?
23. My colleague made inappropriate advances and I want to file a formal complaint.
24. How do I lodge a misconduct complaint about workplace harassment?

### POSH — Caveat Only (should answer with a POSH caveat, not block)
25. What is the company's POSH policy?
26. Is POSH training mandatory for all employees?

### Termination — Caveat Trigger
27. What happens if I am terminated during my probation period?
28. Can I be fired for consistently poor performance reviews?
29. What is the notice period if I am dismissed without cause?

### PIP (Performance Improvement Plan) — Caveat Trigger
30. I have been placed on a PIP — what are my rights and next steps?
31. What happens if I fail my performance improvement plan?

### Legal Action — Caveat Trigger
32. Can I sue the company for wrongful termination?
33. How do I escalate a grievance to the labour court?
34. What is the whistleblower protection policy?

### Salary Reduction — Caveat Trigger
35. Can my salary be cut if I move to a lower band role?
36. Is the company allowed to reduce my CTC without my consent?

### Out-of-Scope / Unknown Intent
37. What is the weather forecast for tomorrow?
38. Can you book a flight ticket to Mumbai for me?

### Ambiguous & Multi-Domain
39. I am on a PIP and want to know if I can take annual leave and also work remotely — what are my options?
40. If I get terminated, does my ESOP vest and can I encash my remaining leave?

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: faiss` | `pip install faiss-cpu` |
| `ModuleNotFoundError: streamlit` | `pip install streamlit` |
| Ollama not found | Install from https://ollama.com — template fallback works without it |
| Index not found | Run `python component_a_hr_indexing.py` first |
| Port 8501 busy | `streamlit run hr_copilot_ui.py --server.port 8502` |
| Answer shows "No relevant policy found" | Run `python component_a_hr_indexing.py` to rebuild the index |

*HR Copilot · Maneesh Kumar *
>>>>>>> 40b43563 (Initial commit)
