"""
============================================================
HR Copilot — Streamlit Web UI
============================================================
Run with:  streamlit run hr_copilot_ui.py
============================================================
"""
import os, sys, io, time, json, contextlib
# Must be set before any HuggingFace/torch imports to prevent semaphore crashes
# when multiple models (NLI + cross-encoder + embeddings) are loaded together
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import streamlit as st
import pandas as pd

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="HR Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Ensure we're in the right working directory ──────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(THIS_DIR)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main header */
.main-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
    padding: 20px 30px;
    border-radius: 12px;
    margin-bottom: 20px;
    color: white;
}
.main-header h1 { margin: 0; font-size: 2rem; }
.main-header p  { margin: 4px 0 0 0; opacity: 0.85; font-size: 0.95rem; }

/* Agent cards */
.agent-card {
    border-left: 4px solid;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    margin: 6px 0;
    background: #f8fafc;
}
.agent-orchestrator { border-color: #7c3aed; }
.agent-policy      { border-color: #0ea5e9; }
.agent-data        { border-color: #10b981; }
.agent-onboarding  { border-color: #f59e0b; }
.agent-compliance  { border-color: #ef4444; }
.agent-synthesizer { border-color: #6366f1; }

/* Metric pills — explicit colors so they work in both light and dark mode */
.metric-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 3px;
}
.metric-pass { background: #d1fae5 !important; color: #065f46 !important; }
.metric-warn { background: #fef3c7 !important; color: #92400e !important; }
.metric-fail { background: #fee2e2 !important; color: #991b1b !important; }

/* Intent badge */
.intent-badge {
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
}

/* Pipeline diagram */
.pipeline-step {
    text-align: center;
    padding: 8px;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 600;
}

/* Scenario button area */
.scenario-category {
    font-weight: 700;
    font-size: 0.85rem;
    color: #374151;
    margin: 12px 0 4px 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Status dot */
.status-dot-green { color: #10b981; font-size: 1.1rem; }
.status-dot-red   { color: #ef4444; font-size: 1.1rem; }
.status-dot-amber { color: #f59e0b; font-size: 1.1rem; }

/* Chat messages */
.chat-user {
    background: #e0e7ff;
    border-radius: 12px 12px 4px 12px;
    padding: 10px 16px;
    margin: 6px 0;
    text-align: right;
}
.chat-bot {
    background: #f1f5f9;
    border-radius: 12px 12px 12px 4px;
    padding: 10px 16px;
    margin: 6px 0;
}

/* Sidebar status */
.sidebar-status {
    background: #1e293b;
    color: #e2e8f0;
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 0.82rem;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
INDEX_DIR = "data/index"
FAITHFULNESS_GATE = 0.80

SCENARIOS = {
    "Leave Policy": [
        "What is the maximum annual leave carry-forward and can it be encashed?",
        "How many sick leave days am I entitled to per year?",
        "What is the maternity leave duration at Enterprise Corp?",
        "Can I combine casual leave and annual leave for a long trip?",
        "What bereavement leave am I entitled to if a parent passes away?",
    ],
    "Compensation & Benefits": [
        "What is the salary band range for a Band 3 Senior Lead?",
        "What is the salary range for a Band 4 Manager and are they ESOP eligible?",
        "What are the tax-exempt components in my monthly salary?",
        "How is variable pay calculated and when is it paid?",
        "What is the performance rating and increment matrix?",
    ],
    "Remote Work": [
        "Can I work from home 3 days per week if I completed probation?",
        "Can I work from outside India for a month this year?",
        "What are the core working hours I must be available during?",
        "I am on probation — how many days must I come to office?",
        "How do I apply for a recurring weekly remote work schedule?",
    ],
    "Onboarding": [
        "What must I complete in my first week as a new employee?",
        "What documents do I need to submit before joining?",
        "What is my 30-60-90 day plan as a new joiner?",
        "Where do I collect my laptop on Day 1?",
        "What training is mandatory in the first 30 days?",
    ],
    "Learning & Development": [
        "What is my annual L&D budget and what courses are eligible?",
        "I want to do an AWS certification — how do I claim the cost?",
        "Does unused L&D budget carry forward to next year?",
        "What internal learning programmes does the company offer?",
        "What is the clawback policy if I leave after getting sponsored for a certification?",
    ],
    "Headcount & Salary Data": [
        "Which department has the highest attrition and what is the total company headcount?",
        "What is the total number of employees in Engineering?",
        "What is the average tenure across all departments?",
        "How many open positions does the company have right now?",
        "What is the headcount in the Sales department?",
    ],
    "Grievance & Compliance": [
        "How do I file a grievance and what is the resolution timeline?",
        "What is the notice period for a Band 4 Manager?",
        "What are the steps in the disciplinary process?",
        "What is Enterprise Corp's POSH policy?",
        "How do I report an ethics violation anonymously?",
    ],
    "Multi-Domain": [
        "I want to understand my L&D budget and remote work options as a Band 2 employee.",
        "What are my leave entitlements and salary band if I am a Band 3 Lead?",
        "As a new joiner at Band 2, what are my WFH rights and first week tasks?",
        "I am joining as a Manager (Band 4) — what are my ESOP, leave, and notice period?",
    ],
}

INTENT_COLORS = {
    "leave_policy":    "#0ea5e9",
    "compensation":    "#10b981",
    "remote_work":     "#8b5cf6",
    "onboarding":      "#f59e0b",
    "learning":        "#ec4899",
    "headcount_data":  "#14b8a6",
    "salary_band":     "#06b6d4",
    "grievance":       "#ef4444",
    "multi_domain":    "#6366f1",
    "unknown":         "#94a3b8",
}

AGENT_ICONS = {
    "OrchestratorAgent":    "🎯",
    "PolicyRAGAgent":       "📚",
    "DataQueryAgent":       "📊",
    "OnboardingAgent":      "🚀",
    "ComplianceGuardAgent": "🛡️",
}

# ── Pipeline caching ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_pipeline():
    from hr_copilot_pipeline import HRCopilotPipeline
    return HRCopilotPipeline(verbose=False)


def index_exists():
    return os.path.exists(f"{INDEX_DIR}/hr_faiss.index")


def sample_data_exists():
    return os.path.exists("data/hr_docs/leave_policy.md")


# ── Helper: run pipeline and capture logs ────────────────────────────────────
def run_query(question: str, use_llm: bool = False):
    """Run the pipeline, capture stdout logs, return (result_dict, log_text).
    Results are cached locally in session state so repeated questions return instantly."""
    # Check local cache first
    if "query_cache" not in st.session_state:
        st.session_state.query_cache = {}
    cache_key = f"{question.strip().lower()}|{use_llm}"
    if cache_key in st.session_state.query_cache:
        cached = st.session_state.query_cache[cache_key]
        return cached["result"], cached["log"]

    pipeline = get_pipeline()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = pipeline.ask(question, use_llm=use_llm)
    log = buf.getvalue()

    # Store in local cache
    st.session_state.query_cache[cache_key] = {"result": result, "log": log}
    return result, log


# ── Helper: metric pill ──────────────────────────────────────────────────────
def metric_pill(label: str, value: float, good_threshold: float, warn_threshold: float = 0.0):
    if value >= good_threshold:
        cls = "metric-pass"
        icon = "✅"
    elif value >= warn_threshold:
        cls = "metric-warn"
        icon = "⚠️"
    else:
        cls = "metric-fail"
        icon = "❌"
    return f'<span class="metric-pill {cls}">{icon} {label}: {value:.2f}</span>'


# ── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🤖 HR Copilot")
        st.markdown("*Multi-Agent Employee Self-Service*")
        st.divider()

        # System status
        st.markdown("### System Status")

        sample_ok = sample_data_exists()
        index_ok  = index_exists()

        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("📄" if sample_ok else "❌")
        with col2:
            st.markdown(f"Sample data {'ready' if sample_ok else '**missing**'}")

        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("🗂️" if index_ok else "❌")
        with col2:
            st.markdown(f"Index {'built' if index_ok else '**not built**'}")

        st.divider()

        # Setup actions
        st.markdown("### Setup")

        if not sample_ok:
            if st.button("📄 Create Sample Data", use_container_width=True, type="primary"):
                with st.spinner("Creating sample HR documents..."):
                    import subprocess
                    subprocess.run([sys.executable, "create_sample_data.py"], check=True)
                st.success("Sample data created!")
                st.rerun()

        if not index_ok:
            st.warning("Index not built. Build it to start answering questions.")
            if st.button("🔨 Build Knowledge Index", use_container_width=True, type="primary"):
                with st.spinner("Building FAISS + BM25 index... (this takes 1-2 min on first run)"):
                    pipeline = get_pipeline()
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        pipeline.build_index()
                st.success("Index built successfully!")
                st.rerun()
        else:
            if st.button("🔄 Rebuild Index", use_container_width=True):
                with st.spinner("Rebuilding index..."):
                    pipeline = get_pipeline()
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        pipeline.build_index()
                st.success("Index rebuilt!")
                st.rerun()

        st.divider()

        st.divider()

        # Agent legend
        st.markdown("### Pipeline Agents")
        for agent, icon in AGENT_ICONS.items():
            st.markdown(f"{icon} `{agent}`")

        st.divider()
        st.caption("MLDS 2026 · Maneesh Kumar & Ravikiran Ravada")


# ── Pipeline diagram ─────────────────────────────────────────────────────────
def render_pipeline_diagram():
    steps = [
        ("User Query", "#94a3b8"),
        ("B: Orchestrate", "#7c3aed"),
        ("C: Retrieve", "#0ea5e9"),
        ("D: Compliance", "#ef4444"),
        ("E: Synthesize", "#10b981"),
        ("Answer", "#1e3a5f"),
    ]
    cols = st.columns(len(steps))
    for col, (label, color) in zip(cols, steps):
        with col:
            st.markdown(
                f'<div style="background:{color};color:white;text-align:center;'
                f'padding:8px 4px;border-radius:8px;font-size:0.78rem;font-weight:700;">'
                f'{label}</div>',
                unsafe_allow_html=True
            )


# ── Render a single query result ─────────────────────────────────────────────
def render_result(result: dict, log: str = ""):
    intent = result.get("intent", "unknown")
    color  = INTENT_COLORS.get(intent, "#94a3b8")

    # Intent + compliance badge
    compliance_html = (
        '<span class="metric-pill metric-pass">✅ Compliance PASS</span>'
        if result["compliance_passed"]
        else '<span class="metric-pill metric-fail">❌ Compliance BLOCKED</span>'
    )
    st.markdown(
        f'<span class="intent-badge" style="background:{color}22;color:{color};">'
        f'🏷️ {intent.replace("_"," ").title()}</span> &nbsp; {compliance_html}',
        unsafe_allow_html=True
    )

    st.markdown("#### Answer")
    with st.container(border=True):
        st.markdown(result["answer"])

    # Caveats
    if result.get("caveats"):
        for c in result["caveats"]:
            st.warning(c)

    # Metrics row
    faith = result["faithfulness"]
    rel   = result["answer_relevancy"]
    prec  = result["context_precision"]
    lat   = result["latency_ms"]

    metrics_html = (
        metric_pill("Faithfulness", faith, FAITHFULNESS_GATE, 0.60) + " " +
        metric_pill("Relevancy", rel, 0.68, 0.50) + " " +
        metric_pill("Precision", prec, 0.70, 0.50) + " " +
        f'<span class="metric-pill" style="background:#e2e8f0 !important;color:#334155 !important;">⏱️ {lat:.0f} ms</span>'
    )
    st.markdown(metrics_html, unsafe_allow_html=True)

    # Sources & agents
    col1, col2 = st.columns(2)
    with col1:
        if result["sources"]:
            st.markdown("**Sources**")
            for s in result["sources"]:
                st.markdown(f"- 📄 `{s}`")
        else:
            st.caption("No document sources cited")
    with col2:
        st.markdown("**Contributing Agents**")
        for a in result["agents"]:
            icon = AGENT_ICONS.get(a, "🤖")
            st.markdown(f"- {icon} `{a}`")

    # Agent trace (expandable)
    if log.strip():
        with st.expander("🔍 Agent Trace (Pipeline Logs)", expanded=False):
            clean_log = log.replace("─","─").replace("✅","✅").replace("⚠️","⚠️").replace("❌","❌")
            st.code(clean_log, language=None)


# ── Tab 1: Chat ──────────────────────────────────────────────────────────────
def render_chat_tab():
    st.markdown("### 💬 Ask an HR Question")

    if not index_exists():
        st.error("Knowledge index not built yet. Use the sidebar to build it first.")
        return

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Input form with Send button (clear_on_submit prevents reprocessing)
    with st.form(key="chat_form", clear_on_submit=True):
        input_col, send_col = st.columns([5, 1])
        with input_col:
            question = st.text_input(
                "Ask anything about leave, salary, onboarding, remote work...",
                key="chat_input",
                label_visibility="collapsed",
                placeholder="Ask anything about leave, salary, onboarding, remote work...",
            )
        with send_col:
            submitted = st.form_submit_button("📨 Send", use_container_width=True, type="primary")

    # Clear button outside the form
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    # Process new question (only when form is submitted with non-empty input)
    if submitted and question.strip():
        with st.spinner("Processing through multi-agent pipeline..."):
            result, log = run_query(question, use_llm=True)
        st.session_state.chat_history.append({
            "question": question,
            "result":   result,
            "log":      log,
        })
        st.rerun()

    # Display history — newest first so older queries are pushed down
    for entry in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant", avatar="🤖"):
            render_result(entry["result"], entry.get("log", ""))


# ── Tab 2: Scenarios ──────────────────────────────────────────────────────────
def render_scenarios_tab():
    st.markdown("### 🎯 Test Scenarios")
    st.markdown("Click any scenario to run it through the full multi-agent pipeline.")

    if not index_exists():
        st.error("Knowledge index not built yet. Use the sidebar to build it first.")
        return

    # Show scenario result if one is selected
    if "scenario_result" in st.session_state and st.session_state.scenario_result:
        entry = st.session_state.scenario_result
        st.divider()
        st.markdown(f"**Q: {entry['question']}**")
        render_result(entry["result"], entry.get("log", ""))
        st.divider()

    # Scenario buttons grouped by category
    for category, questions in SCENARIOS.items():
        with st.expander(f"**{category}** ({len(questions)} scenarios)", expanded=(category == "Leave Policy")):
            cols = st.columns(2)
            for i, q in enumerate(questions):
                col = cols[i % 2]
                with col:
                    if st.button(f"▶ {q[:65]}{'...' if len(q)>65 else ''}",
                                 key=f"scenario_{category}_{i}",
                                 use_container_width=True):
                        with st.spinner(f"Running: {q[:50]}..."):
                            result, log = run_query(q, use_llm=True)
                        st.session_state.scenario_result = {
                            "question": q,
                            "result":   result,
                            "log":      log,
                        }
                        st.rerun()


# ── Tab 3: Eval Suite ────────────────────────────────────────────────────────
def render_eval_tab():
    st.markdown("### 📊 Evaluation Suite")
    st.markdown(
        "Runs 6 representative HR questions and computes RAGAS quality metrics. "
        f"CI/CD gate requires **Faithfulness ≥ {FAITHFULNESS_GATE}**."
    )

    if not index_exists():
        st.error("Knowledge index not built yet. Use the sidebar to build it first.")
        return

    EVAL_QUESTIONS = [
        ("Leave", "What is the maximum annual leave carry-forward and can it be encashed?"),
        ("Compensation", "What is the salary band range for a Band 3 Senior Lead?"),
        ("Remote Work", "Can I work from home 3 days per week if I completed probation?"),
        ("Onboarding", "What must I complete in my first week as a new employee?"),
        ("Headcount", "Which department has the highest attrition and what is the total company headcount?"),
        ("Multi-Domain", "I want to understand my L&D budget and remote work options as a Band 2 employee."),
    ]

    # Show existing report if available
    report_path = "data/eval/eval_suite_report.json"
    if os.path.exists(report_path):
        try:
            with open(report_path) as f:
                report = json.load(f)
            st.markdown("#### Last Evaluation Report")
            gate = report.get("gate_passed", False)
            if gate:
                st.success(f"✅ CI/CD Gate PASSED — Avg Faithfulness: {report['avg_faithfulness']:.3f}")
            else:
                st.error(f"❌ CI/CD Gate FAILED — Avg Faithfulness: {report['avg_faithfulness']:.3f} (need ≥ {FAITHFULNESS_GATE})")

            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Faithfulness", f"{report['avg_faithfulness']:.3f}")
            col2.metric("Avg Relevancy",    f"{report['avg_relevancy']:.3f}")
            col3.metric("Questions Run",    str(len(report.get("results", []))))

            if report.get("results"):
                rows = []
                for r in report["results"]:
                    rows.append({
                        "Question": r["question"][:60] + ("..." if len(r["question"])>60 else ""),
                        "Intent":   r["intent"],
                        "Faithfulness": round(r["faithfulness"], 3),
                        "Relevancy":    round(r["answer_relevancy"], 3),
                        "Precision":    round(r["context_precision"], 3),
                        "Latency (ms)": round(r["latency_ms"]),
                        "Compliance":   "✅" if r["compliance_passed"] else "❌",
                    })
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
            st.divider()
        except Exception as e:
            st.warning(f"Could not load existing report: {e}")

    # Questions preview
    st.markdown("#### Evaluation Questions")
    for domain, q in EVAL_QUESTIONS:
        st.markdown(f"- **{domain}**: {q}")

    # Run button
    if st.button("🚀 Run Full Evaluation Suite", type="primary", use_container_width=True):
        pipeline = get_pipeline()
        results  = []
        progress = st.progress(0, text="Starting evaluation...")
        status   = st.empty()
        results_placeholder = st.container()

        for i, (domain, q) in enumerate(EVAL_QUESTIONS):
            status.markdown(f"**[{i+1}/{len(EVAL_QUESTIONS)}]** Running: *{q[:60]}...*")
            progress.progress((i) / len(EVAL_QUESTIONS), text=f"Running question {i+1}/{len(EVAL_QUESTIONS)}...")

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                r = pipeline.ask(q, use_llm=st.session_state.get("use_llm", False))
            results.append(r)

            with results_placeholder:
                st.markdown(f"**✅ {domain}**: Faith={r['faithfulness']:.3f} | Rel={r['answer_relevancy']:.3f} | {r['latency_ms']:.0f}ms")

        progress.progress(1.0, text="Evaluation complete!")
        status.empty()

        # Aggregate
        avg_faith = sum(r["faithfulness"] for r in results) / len(results)
        avg_rel   = sum(r["answer_relevancy"] for r in results) / len(results)
        gate_pass = avg_faith >= FAITHFULNESS_GATE

        # Save report
        os.makedirs("data/eval", exist_ok=True)
        with open(report_path, "w") as f:
            json.dump({
                "gate_passed":      gate_pass,
                "avg_faithfulness": round(avg_faith, 3),
                "avg_relevancy":    round(avg_rel, 3),
                "faithfulness_gate": FAITHFULNESS_GATE,
                "results":          results,
            }, f, indent=2)

        if gate_pass:
            st.success(f"✅ CI/CD Gate PASSED — Avg Faithfulness: {avg_faith:.3f}")
        else:
            st.error(f"❌ CI/CD Gate FAILED — Avg Faithfulness: {avg_faith:.3f} (need ≥ {FAITHFULNESS_GATE})")

        st.rerun()


# ── Tab 4: Knowledge Base ─────────────────────────────────────────────────────
def render_knowledge_tab():
    st.markdown("### 📚 Knowledge Base")

    col1, col2 = st.columns(2)

    # HR Policy documents
    with col1:
        st.markdown("#### HR Policy Documents")
        docs_dir = "data/hr_docs"
        if os.path.exists(docs_dir):
            docs = [f for f in os.listdir(docs_dir) if f.endswith(".md")]
            if docs:
                doc_choice = st.selectbox("Select document to preview:", sorted(docs))
                if doc_choice:
                    with open(os.path.join(docs_dir, doc_choice), encoding="utf-8") as f:
                        content = f.read()
                    with st.expander(f"📄 {doc_choice}", expanded=True):
                        st.markdown(content)
            else:
                st.info("No policy documents found. Create sample data from the sidebar.")
        else:
            st.info("data/hr_docs/ directory not found.")

    # Structured data
    with col2:
        st.markdown("#### Structured HR Data")

        # Salary bands
        salary_path = "data/hr_structured/salary_bands.json"
        if os.path.exists(salary_path):
            with open(salary_path) as f:
                salary_data = json.load(f)
            bands = salary_data.get("bands", [])
            if bands:
                df = pd.DataFrame(bands)
                df["min_ctc"] = df["min_ctc"].apply(lambda x: f"₹{x:,.0f}")
                df["max_ctc"] = df["max_ctc"].apply(lambda x: f"₹{x:,.0f}")
                df.columns = ["Band", "Title", "Min CTC", "Max CTC", "Notice (months)", "ESOP Eligible"]
                st.markdown("**Salary Bands**")
                st.dataframe(df, use_container_width=True, hide_index=True)

        # Headcount
        headcount_path = "data/hr_structured/headcount.csv"
        if os.path.exists(headcount_path):
            df_hc = pd.read_csv(headcount_path)
            df_hc.columns = [c.replace("_", " ").title() for c in df_hc.columns]
            st.markdown("**Departmental Headcount**")
            st.dataframe(df_hc, use_container_width=True, hide_index=True)

            # Quick chart
            st.bar_chart(
                df_hc.set_index("Department")["Headcount"],
                use_container_width=True,
                color="#2d6a9f",
            )

    # Index stats
    if index_exists():
        st.divider()
        st.markdown("#### Index Statistics")
        chunks_path = f"{INDEX_DIR}/hr_chunks.json"
        if os.path.exists(chunks_path):
            with open(chunks_path) as f:
                chunks = json.load(f)
            by_category = {}
            for c in chunks:
                cat = c.get("category", "unknown")
                by_category[cat] = by_category.get(cat, 0) + 1

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Chunks Indexed", len(chunks))
            col2.metric("HR Categories", len(by_category))
            col3.metric("Index Location", INDEX_DIR)

            df_cat = pd.DataFrame([{"Category": k, "Chunks": v}
                                   for k, v in sorted(by_category.items(), key=lambda x: -x[1])])
            st.dataframe(df_cat, use_container_width=True, hide_index=True)


# ── Tab 5: Architecture ───────────────────────────────────────────────────────
def render_architecture_tab():
    st.markdown("### 🏗️ Multi-Agent Architecture")

    st.markdown("""
    HR Copilot uses a **5-component multi-agent pipeline** where each agent is a specialist:
    """)

    render_pipeline_diagram()

    st.divider()

    agents = [
        {
            "id": "B",
            "name": "OrchestratorAgent",
            "icon": "🎯",
            "color": "#7c3aed",
            "role": "Intent Classifier & Router",
            "desc": "Classifies the employee's question into 9 HR intent categories and decides which specialist agents to invoke. Decomposes complex multi-part questions into sub-queries.",
            "intents": ["leave_policy", "compensation", "remote_work", "onboarding", "learning", "headcount_data", "salary_band", "grievance", "multi_domain"],
            "output": "HRQueryPlan",
        },
        {
            "id": "C",
            "name": "PolicyRAGAgent",
            "icon": "📚",
            "color": "#0ea5e9",
            "role": "Hybrid Policy Retriever",
            "desc": "Retrieves relevant policy text using FAISS vector search + BM25 keyword search, then fuses results with Reciprocal Rank Fusion (RRF) for best-of-both retrieval.",
            "intents": ["leave_policy", "compensation", "remote_work", "onboarding", "learning", "grievance"],
            "output": "AgentResponse + RetrievedChunks",
        },
        {
            "id": "C",
            "name": "DataQueryAgent",
            "icon": "📊",
            "color": "#10b981",
            "role": "Structured Data Lookup",
            "desc": "Performs exact lookups on structured HR data: salary bands (JSON) and departmental headcount (CSV). Returns precise numbers, not approximations.",
            "intents": ["salary_band", "headcount_data", "compensation"],
            "output": "AgentResponse",
        },
        {
            "id": "C",
            "name": "OnboardingAgent",
            "icon": "🚀",
            "color": "#f59e0b",
            "role": "New Joiner Specialist",
            "desc": "Combines a hardcoded onboarding checklist (pre-joining, Day 1, Week 1, 30-60-90 plan) with RAG retrieval scoped to onboarding policy documents.",
            "intents": ["onboarding"],
            "output": "AgentResponse",
        },
        {
            "id": "D",
            "name": "ComplianceGuardAgent",
            "icon": "🛡️",
            "color": "#ef4444",
            "role": "Compliance & Quality Gate",
            "desc": "Three-stage pipeline: (1) Cross-encoder reranking for precision, (2) NLI fact-checking to verify answers are grounded in evidence, (3) Legal scan for POSH/termination/PIP topics with mandatory caveats.",
            "intents": ["all"],
            "output": "ComplianceCheckResult + verified chunks",
        },
        {
            "id": "E",
            "name": "ResponseSynthesizerAgent",
            "icon": "✍️",
            "color": "#6366f1",
            "role": "Merger + RAGAS Evaluator",
            "desc": "Merges responses from all active agents, applies intent-specific formatting (bullets for leave, tables for salary), adds compliance caveats, then evaluates quality with RAGAS (Faithfulness, Relevancy, Precision).",
            "intents": ["all"],
            "output": "FinalHRResponse",
        },
    ]

    for agent in agents:
        with st.expander(f"{agent['icon']} **[{agent['id']}] {agent['name']}** — {agent['role']}", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**Description:** {agent['desc']}")
                st.markdown(f"**Output type:** `{agent['output']}`")
            with col2:
                st.markdown("**Handles intents:**")
                intents = agent["intents"] if agent["intents"] != ["all"] else ["All HR intents"]
                for i in intents:
                    color = INTENT_COLORS.get(i, "#94a3b8")
                    st.markdown(
                        f'<span class="intent-badge" style="background:{color}22;color:{color};">{i}</span>',
                        unsafe_allow_html=True
                    )

    st.divider()
    st.markdown("### 📐 Data Contracts Between Agents")
    st.code("""
HRQueryPlan        → OrchestratorAgent output (intent, agents, sub_queries, priority_docs)
AgentResponse      → Each specialist's answer (text, sources, confidence, chunks_used)
RetrievedChunk     → Single knowledge piece (text, rrf_score, rerank_score, entail_score)
ComplianceCheckResult → Guard verdict (passes, flags, corrected_answer)
FinalHRResponse    → Employee-facing answer (answer, sources, RAGAS scores, caveats)
    """, language="python")

    st.markdown("### 🎯 RAGAS Quality Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Faithfulness ≥ 0.80** (CI/CD Gate)")
        st.markdown("Fraction of answer sentences supported by retrieved evidence. Primary hallucination detector.")
    with col2:
        st.markdown("**Answer Relevancy ≥ 0.68** (Warning)")
        st.markdown("Cosine similarity between question and answer embeddings. Measures topical alignment.")
    with col3:
        st.markdown("**Context Precision**")
        st.markdown("Fraction of retrieved chunks that contributed to the answer. Low = noisy retrieval.")


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 HR Copilot</h1>
        <p>Multi-Agent Employee Self-Service System · MLDS 2026 · Maneesh Kumar & Ravikiran Ravada</p>
    </div>
    """, unsafe_allow_html=True)

    render_sidebar()

    # Quick status banner
    if not sample_data_exists():
        st.warning("⚠️ Sample HR data not found. Click **Create Sample Data** in the sidebar.")
    elif not index_exists():
        st.warning("⚠️ Knowledge index not built. Click **Build Knowledge Index** in the sidebar.")
    else:
        st.success("✅ System ready — knowledge base indexed and all agents available.")

    # Pipeline overview
    with st.expander("🔄 Pipeline Overview", expanded=False):
        render_pipeline_diagram()
        st.caption("Each query flows through 5 specialized agents: Orchestrate → Retrieve → Comply → Synthesize → Evaluate")

    st.divider()

    # Tabs
    tabs = st.tabs(["💬 Chat", "🎯 Scenarios", "📊 Eval Suite", "📚 Knowledge Base", "🏗️ Architecture"])

    with tabs[0]:
        render_chat_tab()

    with tabs[1]:
        render_scenarios_tab()

    with tabs[2]:
        render_eval_tab()

    with tabs[3]:
        render_knowledge_tab()

    with tabs[4]:
        render_architecture_tab()


if __name__ == "__main__":
    main()
