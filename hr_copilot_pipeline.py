"""
============================================================
HR COPILOT — End-to-End Multi-Agent Pipeline
============================================================
HR Copilot · Maneesh Kumar 

This is the orchestration layer that wires all 5 components
together into a production-grade multi-agent pipeline.

────────────────────────────────────────────────────────────
HOW THE MULTI-AGENT FRAMEWORK IS USED HERE
────────────────────────────────────────────────────────────
  1. AgentRegistry   — stores all specialist agents centrally
  2. OrchestratorAgent.plan() — decides who to call and how
  3. ParallelAgentExecutor.execute() — runs specialist agents
     concurrently (PolicyRAG + DataQuery + Onboarding in parallel)
  4. ComplianceGuardAgent — sequential gate (order matters)
  5. ResponseSynthesizerAgent — merges all outputs

────────────────────────────────────────────────────────────
PIPELINE FLOW
────────────────────────────────────────────────────────────
  Employee Question
    → [A] Knowledge Base (FAISS + BM25 index, built once)
    → [B] OrchestratorAgent     (intent classify + route)
    → [C] ParallelAgentExecutor (fan-out to specialist agents):
            ├─ PolicyRAGAgent     (hybrid retrieval)
            ├─ DataQueryAgent     (structured data)
            └─ OnboardingAgent    (new joiner checklist)
    → [D] ComplianceGuardAgent  (rerank + fact-check + legal)
    → [E] ResponseSynthesizerAgent (merge + format + cite)
    → [E] RAGAS Evaluation      (quality gate)

────────────────────────────────────────────────────────────
CLI USAGE
────────────────────────────────────────────────────────────
  python3 hr_copilot_pipeline.py --build-index    # First time
  python3 hr_copilot_pipeline.py                  # Interactive
  python3 hr_copilot_pipeline.py --eval           # Eval suite
  python3 hr_copilot_pipeline.py --question "..." # Single Q
  python3 hr_copilot_pipeline.py --eval --gate    # CI/CD gate
============================================================
"""
import os, sys, time, json, argparse, textwrap
from typing import List, Optional
from dataclasses import asdict

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    CY=Fore.CYAN; CG=Fore.GREEN; CY2=Fore.YELLOW; CR=Fore.RED
    CB=Style.BRIGHT; CR2=Style.RESET_ALL
except ImportError:
    CY=CG=CY2=CR=CB=CR2=""

def hdr(t, c=None): print(f"\n{c or CY}{CB}{'─'*60}\n  {t}\n{'─'*60}{CR2}")
def ok(t):   print(f"  {CG}✅  {t}{CR2}")
def warn(t): print(f"  {CY2}⚠️   {t}{CR2}")
def err(t):  print(f"  {CR}❌  {t}{CR2}")
def info(t): print(f"  {CY}ℹ️   {t}{CR2}")

INDEX_DIR        = "data/index"
FAITHFULNESS_GATE = 0.80


class HRCopilotPipeline:
    """
    Wires all 5 HR Copilot components into a multi-agent pipeline.

    Key design decisions:
      • Lazy loading: models loaded only on first ask() call.
      • AgentRegistry: all specialist agents registered centrally.
      • ParallelAgentExecutor: specialist agents run concurrently.
      • Sequential gates: compliance and synthesis run after retrieval.

    Usage:
        pipeline = HRCopilotPipeline()
        result   = pipeline.ask("How many leave days can I carry forward?")
        pipeline.print_result(result)
    """

    def __init__(self, verbose: bool = True):
        self.verbose  = verbose
        self._loaded  = False

        # Component instances (lazy-loaded on first query)
        self._chunks = self._faiss = self._bm25 = self._embed = None
        self._policy = self._data = self._onboarding = self._compliance = None
        self._nli    = None

        # Multi-agent framework objects
        self._registry = None
        self._executor = None
        self._orchestrator = None

    def _ensure_loaded(self):
        """
        Lazy-load all agents and register them in the AgentRegistry.
        Called automatically on the first ask() invocation.
        """
        if self._loaded:
            return

        from component_a_hr_indexing import load_index
        from component_b_orchestrator_agent import OrchestratorAgent
        from component_c_policy_data_agents import PolicyRAGAgent, DataQueryAgent
        from component_d_compliance_guard import ComplianceGuardAgent, OnboardingAgent
        from agent_framework import AgentRegistry, ParallelAgentExecutor

        hdr("Loading HR Copilot — Multi-Agent Framework", CY)

        # ── Step 1: Build or load knowledge base ──────────────────────────
        if not os.path.exists(f"{INDEX_DIR}/hr_faiss.index"):
            warn("Index not found — building now...")
            self.build_index()

        self._chunks, self._faiss, self._bm25, self._embed = load_index(INDEX_DIR)
        ok(f"Knowledge base: {len(self._chunks)} chunks indexed")

        # ── Step 2: Instantiate all agents ────────────────────────────────
        self._policy     = PolicyRAGAgent(self._chunks, self._faiss, self._bm25, self._embed)
        self._data       = DataQueryAgent()
        self._compliance = ComplianceGuardAgent()
        self._onboarding = OnboardingAgent(self._policy)
        self._nli        = self._compliance._load_nli()

        # ── Step 3: Register specialist agents in AgentRegistry ───────────
        #   The registry is the central service discovery layer.
        #   ParallelAgentExecutor looks up agents here by AgentName enum.
        self._registry = AgentRegistry()
        self._registry.register(self._policy)      # PolicyRAGAgent
        self._registry.register(self._data)        # DataQueryAgent
        self._registry.register(self._onboarding)  # OnboardingAgent

        # ── Step 4: Create parallel executor and orchestrator ─────────────
        self._executor     = ParallelAgentExecutor(self._registry, max_workers=4)
        self._orchestrator = OrchestratorAgent()

        ok(f"Registry: {self._registry}")
        ok("Multi-agent framework ready — agents will run in parallel")

        self._loaded = True

    def build_index(self, docs_dir: str = "data/hr_docs"):
        """Rebuild the FAISS + BM25 knowledge base index from HR policy docs."""
        from component_a_hr_indexing import (
            load_hr_documents, chunk_all_documents,
            generate_embeddings, build_faiss_index, build_bm25_index, save_index
        )
        hdr("Building HR Knowledge Base Index", CY)
        docs   = load_hr_documents(docs_dir)
        chunks = chunk_all_documents(docs)
        vecs, _ = generate_embeddings(chunks)
        fi      = build_faiss_index(vecs)
        bm25    = build_bm25_index(chunks)
        save_index(chunks, fi, bm25)
        ok(f"Index built: {len(chunks)} chunks from {len(docs)} HR policy documents")
        self._loaded = False   # Force re-init on next ask()

    def ask(self, question: str, use_llm: bool = True) -> dict:
        """
        Run the complete multi-agent pipeline for one HR question.

        Pipeline stages:
          B → Plan:      OrchestratorAgent classifies intent
          C → Retrieve:  ParallelAgentExecutor fans out to specialists
          D → Guard:     ComplianceGuardAgent reranks + fact-checks
          E → Synthesize: ResponseSynthesizerAgent merges + formats
          E → Evaluate:  RAGAS metrics (faithfulness, relevancy, precision)

        Returns:
            dict with keys: question, answer, sources, agents,
            intent, compliance_passed, caveats, faithfulness,
            answer_relevancy, context_precision, latency_ms,
            agent_trace (per-agent timing)
        """
        self._ensure_loaded()

        from component_d_compliance_guard import MANDATORY_CAVEATS
        from component_e_response_synthesizer import (
            synthesizer_agent, evaluate_response
        )
        from hr_data_models import AgentName

        t0 = time.time()

        # ── B: Plan — OrchestratorAgent ───────────────────────────────────
        hdr("B: OrchestratorAgent — Intent Classification & Routing", CY)
        plan = self._orchestrator.plan(question, use_llm=use_llm)
        ok(f"Intent:  {plan.intent.value}")
        ok(f"Agents:  {[a.value for a in plan.agents_to_invoke]}")
        ok(f"Sub-Qs:  {len(plan.sub_queries)}")
        for i, q in enumerate(plan.sub_queries, 1):
            info(f"  {i}. {q}")

        # ── C: Retrieve — ParallelAgentExecutor fan-out ───────────────────
        hdr("C: Specialist Agents — Parallel Execution (Fan-Out)", CY)
        responses, agent_tasks = self._executor.execute(
            plan=plan,
            agent_names=plan.agents_to_invoke,
        )
        self._executor.print_execution_summary(agent_tasks)

        # Also collect raw chunks from PolicyRAGAgent for compliance
        retrieved_all = []
        if AgentName.POLICY_RAG in plan.agents_to_invoke:
            for sq in plan.sub_queries:
                retrieved_all.extend(self._policy.retrieve(sq, plan))

        ok(f"Specialist agents completed: {len(responses)} response(s)")

        # ── D: Guard — ComplianceGuardAgent ──────────────────────────────
        hdr("D: ComplianceGuardAgent — Rerank + Fact-Check + Legal Scan", CY)
        verified, comp_result = self._compliance.run(
            question, retrieved_all,
            responses[0] if responses else None
        )
        ok(f"Verified chunks: {len(verified)} | Compliance: {comp_result.passes} | Flags: {comp_result.flags}")

        # ── E: Synthesize + Evaluate ──────────────────────────────────────
        hdr("E: ResponseSynthesizerAgent — Merge + Format + RAGAS Eval", CY)
        final = synthesizer_agent(question, plan, responses, verified, comp_result)
        final = evaluate_response(final, verified, self._nli, self._embed)
        final.latency_ms = (time.time() - t0) * 1000

        # Build agent trace for UI / debugging
        agent_trace = [
            {
                "agent":      t.agent_name.value,
                "status":     t.status.value,
                "latency_ms": round(t.latency_ms),
            }
            for t in agent_tasks
        ]

        return {
            "question":          final.question,
            "answer":            final.answer,
            "sources":           final.sources,
            "agents":            final.agents_contributed,
            "intent":            final.intent,
            "compliance_passed": final.compliance_passed,
            "caveats":           final.caveats,
            "faithfulness":      round(final.faithfulness, 3),
            "answer_relevancy":  round(final.answer_relevancy, 3),
            "context_precision": round(final.context_precision, 3),
            "latency_ms":        round(final.latency_ms),
            "agent_trace":       agent_trace,
        }

    def print_result(self, result: dict):
        """Pretty-print the final HR Copilot response to the terminal."""
        hdr("HR COPILOT RESPONSE", CG)
        print(f"\n  {CB}Q: {result['question']}{CR2}\n")

        for para in result["answer"].split("\n"):
            if para.strip():
                print(textwrap.fill(
                    para.strip(), 72,
                    initial_indent="  ", subsequent_indent="    "
                ))

        print(f"\n  {CY}Sources:{CR2}  {', '.join(result['sources']) or 'none'}")
        print(f"  {CY}Agents:{CR2}   {', '.join(result['agents'])}")
        print(f"  {CY}Intent:{CR2}   {result['intent']}")
        if result["caveats"]:
            print(f"  {CY2}Caveats:{CR2}  {result['caveats']}")

        # ── RAGAS scores ──────────────────────────────────────────────────
        faith = result["faithfulness"]
        rel   = result["answer_relevancy"]
        prec  = result["context_precision"]
        lat   = result["latency_ms"]

        print(f"\n  {CY}Quality Metrics (RAGAS):{CR2}")
        print(f"    Faithfulness:      {faith:.2f}  {'✅' if faith >= FAITHFULNESS_GATE else '❌'}")
        print(f"    Answer Relevancy:  {rel:.2f}  {'✅' if rel >= 0.68 else '⚠️'}")
        print(f"    Context Precision: {prec:.2f}")
        print(f"    Latency:           {lat} ms")
        print(f"    Compliance:        {'✅ PASS' if result['compliance_passed'] else '❌ BLOCKED'}")

        # ── Per-agent parallel execution trace ────────────────────────────
        if result.get("agent_trace"):
            print(f"\n  {CY}Parallel Agent Trace:{CR2}")
            for t in result["agent_trace"]:
                icon = "✅" if t["status"] == "completed" else "❌"
                print(f"    {icon}  {t['agent']:22s}  {t['latency_ms']:>5}ms")

    def run_eval_suite(self, gate: bool = False) -> bool:
        """Run the RAGAS evaluation suite against 6 predefined HR questions."""
        questions = [
            "What is the maximum annual leave carry-forward and can it be encashed?",
            "What is the salary band range for a Band 3 Senior Lead?",
            "Can I work from home 3 days per week if I completed probation?",
            "What must I complete in my first week as a new employee?",
            "Which department has the highest attrition and what is the total company headcount?",
            "I want to understand my L&D budget and remote work options as a Band 2 employee.",
        ]

        hdr(f"EVALUATION SUITE — {len(questions)} HR Scenarios", CY)
        results = []
        for i, q in enumerate(questions, 1):
            print(f"\n  [{i}/{len(questions)}] {q}")
            r = self.ask(q, use_llm=False)
            self.print_result(r)
            results.append(r)

        avg_faith = sum(r["faithfulness"]     for r in results) / len(results)
        avg_rel   = sum(r["answer_relevancy"] for r in results) / len(results)
        avg_lat   = sum(r["latency_ms"]       for r in results) / len(results)
        passed    = avg_faith >= FAITHFULNESS_GATE

        hdr("EVALUATION SUMMARY", CG if passed else CR)
        print(f"  Questions:         {len(results)}")
        print(f"  Avg Faithfulness:  {avg_faith:.3f}  (gate: {FAITHFULNESS_GATE})")
        print(f"  Avg Relevancy:     {avg_rel:.3f}")
        print(f"  Avg Latency:       {avg_lat:.0f} ms")
        print(f"  CI/CD Gate:        {'✅ PASS' if passed else '❌ BLOCKED — deployment prevented'}")

        os.makedirs("data/eval", exist_ok=True)
        with open("data/eval/eval_suite_report.json", "w") as f:
            json.dump({
                "gate_passed":      passed,
                "avg_faithfulness": round(avg_faith, 3),
                "avg_relevancy":    round(avg_rel, 3),
                "faithfulness_gate": FAITHFULNESS_GATE,
                "results":          results,
            }, f, indent=2)
        ok("Report saved: data/eval/eval_suite_report.json")

        if gate:
            sys.exit(0 if passed else 1)
        return passed

    def interactive(self):
        """Interactive Q&A terminal loop."""
        hdr("HR COPILOT — Interactive Mode (Ctrl+C to exit)", CG)
        print("  Ask any HR question. Examples:")
        print("  • 'How many leave days can I carry forward?'")
        print("  • 'What is the salary band for a manager?'")
        print("  • 'What do I do on my first day?'")
        print("  • 'Can I work from home 4 days a week?'")
        print("  • 'What is the total headcount?'")

        while True:
            try:
                q = input(f"\n  {CB}Your question:{CR2} ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n  Goodbye!")
                break
            if not q or q.lower() in ("exit", "quit", "q"):
                break
            result = self.ask(q, use_llm=True)
            self.print_result(result)


def main():
    parser = argparse.ArgumentParser(
        description="HR Copilot — Multi-Agent Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 hr_copilot_pipeline.py --build-index
  python3 hr_copilot_pipeline.py --question "What is my leave carry-forward?"
  python3 hr_copilot_pipeline.py --eval
  python3 hr_copilot_pipeline.py --eval --gate
        """,
    )
    parser.add_argument("--build-index", action="store_true",
                        help="Rebuild FAISS + BM25 knowledge base")
    parser.add_argument("--eval",        action="store_true",
                        help="Run RAGAS evaluation suite")
    parser.add_argument("--gate",        action="store_true",
                        help="Exit code 1 if faithfulness < 0.80")
    parser.add_argument("--question",    type=str, default=None,
                        help="Single HR question to answer")
    parser.add_argument("--quiet",       action="store_true",
                        help="Suppress verbose output")
    args = parser.parse_args()

    print(f"\n{CB}{CY}")
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║   HR Copilot — Multi-Agent Employee Self-Service    ║")
    print("  ║   Multi-Agent Framework · Maneesh Kumar             ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print(f"  Framework: AgentRegistry + ParallelAgentExecutor{CR2}")

    pipeline = HRCopilotPipeline(verbose=not args.quiet)

    if args.build_index:
        pipeline.build_index()
        ok("Index built. Run without --build-index to start.")
        return

    if args.eval:
        pipeline.run_eval_suite(gate=args.gate)
        return

    if args.question:
        pipeline._ensure_loaded()
        result = pipeline.ask(args.question, use_llm=True)
        pipeline.print_result(result)
        if args.gate:
            sys.exit(0 if result["faithfulness"] >= FAITHFULNESS_GATE else 1)
        return

    # Default: 3 demo queries then interactive mode
    pipeline._ensure_loaded()
    demos = [
        "What is the carry-forward limit for annual leave?",
        "What salary band range does a Band 4 manager fall in?",
        "I am joining next week — what documents do I need?",
    ]
    hdr("DEMO QUERIES (parallel agents running for each)", CY)
    for q in demos:
        result = pipeline.ask(q, use_llm=False)
        pipeline.print_result(result)

    pipeline.interactive()


if __name__ == "__main__":
    main()
