"""
============================================================
HR COPILOT — End-to-End Multi-Agent Pipeline
============================================================
HR Copilot · MLDS 2026 · Maneesh Kumar & Ravikiran Ravada

Usage:
  python3 hr_copilot_pipeline.py --build-index    # First time
  python3 hr_copilot_pipeline.py                  # Interactive
  python3 hr_copilot_pipeline.py --eval           # Eval suite
  python3 hr_copilot_pipeline.py --question "..." # Single Q

Multi-Agent Pipeline:
  User Query
    → [B] OrchestratorAgent     (intent classify + route)
    → [C] PolicyRAGAgent        (hybrid retrieval)
    → [C] DataQueryAgent        (structured data)
    → [C] OnboardingAgent       (new joiner specialist)
    → [D] ComplianceGuardAgent  (rerank + fact-check + legal)
    → [E] ResponseSynthesizerAgent (merge + format + cite)
    → [E] RAGAS Evaluation      (quality gate)
============================================================
"""
import os, sys, time, json, argparse, textwrap
from typing import List, Optional
from dataclasses import asdict

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    CY=Fore.CYAN; CG=Fore.GREEN; CY2=Fore.YELLOW; CR=Fore.RED; CB=Style.BRIGHT; CR2=Style.RESET_ALL
except ImportError:
    CY=CG=CY2=CR=CB=CR2=""

def hdr(t,c=None): print(f"\n{c or CY}{CB}{'─'*60}\n  {t}\n{'─'*60}{CR2}")
def ok(t):  print(f"  {CG}✅  {t}{CR2}")
def warn(t): print(f"  {CY2}⚠️   {t}{CR2}")
def err(t):  print(f"  {CR}❌  {t}{CR2}")
def info(t): print(f"  {CY}ℹ️   {t}{CR2}")

INDEX_DIR = "data/index"
FAITHFULNESS_GATE = 0.80


class HRCopilotPipeline:
    """
    Orchestrates the full 5-component multi-agent HR Copilot pipeline.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._loaded = False
        # Agent instances (lazy-loaded)
        self._chunks = self._faiss = self._bm25 = self._embed = None
        self._policy = self._data = self._onboarding = self._compliance = None
        self._nli = None

    def _ensure_loaded(self):
        if self._loaded:
            return
        from component_a_hr_indexing import load_index
        from component_c_policy_data_agents import PolicyRAGAgent, DataQueryAgent
        from component_d_compliance_guard import ComplianceGuardAgent, OnboardingAgent

        hdr("Loading HR Copilot Components", CY)

        if not os.path.exists(f"{INDEX_DIR}/hr_faiss.index"):
            warn("Index not found — building now...")
            self.build_index()

        self._chunks, self._faiss, self._bm25, self._embed = load_index(INDEX_DIR)
        ok(f"Knowledge base: {len(self._chunks)} chunks indexed")

        self._policy     = PolicyRAGAgent(self._chunks, self._faiss, self._bm25, self._embed)
        self._data       = DataQueryAgent()
        self._compliance = ComplianceGuardAgent()
        self._onboarding = OnboardingAgent(self._policy)
        self._nli        = self._compliance._load_nli()
        ok("All 4 specialist agents ready")

        self._loaded = True

    def build_index(self, docs_dir: str = "data/hr_docs"):
        from component_a_hr_indexing import (
            load_hr_documents, chunk_all_documents,
            generate_embeddings, build_faiss_index, build_bm25_index, save_index
        )
        hdr("Building HR Knowledge Base Index", CY)
        docs   = load_hr_documents(docs_dir)
        chunks = chunk_all_documents(docs)
        vecs, _= generate_embeddings(chunks)
        fi     = build_faiss_index(vecs)
        bm25   = build_bm25_index(chunks)
        save_index(chunks, fi, bm25)
        ok(f"Index built: {len(chunks)} chunks from {len(docs)} HR policy documents")
        self._loaded = False  # Force reload

    def ask(self, question: str, use_llm: bool = True) -> dict:
        """
        Run the complete multi-agent pipeline for one question.
        Returns a dict with answer, sources, scores, and agent trace.
        """
        self._ensure_loaded()

        from component_b_orchestrator_agent import orchestrator_agent
        from component_d_compliance_guard import MANDATORY_CAVEATS
        from component_e_response_synthesizer import (
            synthesizer_agent, evaluate_response
        )
        from hr_data_models import AgentName

        t0 = time.time()

        # ── B: Orchestrate ────────────────────────────────────────────────────
        hdr("B: OrchestratorAgent — Routing", CY)
        plan = orchestrator_agent(question, use_llm=use_llm)
        ok(f"Intent: {plan.intent.value} | Agents: {[a.value for a in plan.agents_to_invoke]}")

        # ── C: Specialist Agents ─────────────────────────────────────────────
        hdr("C: Specialist Agents — Retrieving", CY)
        responses, retrieved_all = [], []

        if AgentName.POLICY_RAG in plan.agents_to_invoke:
            r = self._policy.run(plan)
            responses.append(r)
            for sq in plan.sub_queries:
                retrieved_all.extend(self._policy.retrieve(sq, plan))
            ok(f"PolicyRAGAgent: {r.chunks_used} chunks")

        if AgentName.DATA_QUERY in plan.agents_to_invoke or plan.needs_structured:
            r = self._data.run(plan)
            responses.append(r)
            ok(f"DataQueryAgent: structured={r.data_used}")

        if AgentName.ONBOARDING in plan.agents_to_invoke:
            r = self._onboarding.run(plan)
            responses.append(r)
            ok(f"OnboardingAgent: sources={r.sources}")

        # ── D: Compliance Guard ───────────────────────────────────────────────
        hdr("D: ComplianceGuardAgent — Verifying", CY)
        verified, comp_result = self._compliance.run(
            question, retrieved_all,
            responses[0] if responses else None
        )
        ok(f"Verified: {len(verified)} chunks | Compliance: {comp_result.passes} | Flags: {comp_result.flags}")

        # ── E: Synthesize + Evaluate ──────────────────────────────────────────
        hdr("E: ResponseSynthesizerAgent — Merging + Evaluating", CY)
        final = synthesizer_agent(question, plan, responses, verified, comp_result)
        final = evaluate_response(final, verified, self._nli, self._embed)
        final.latency_ms = (time.time() - t0) * 1000

        return {
            "question":           final.question,
            "answer":             final.answer,
            "sources":            final.sources,
            "agents":             final.agents_contributed,
            "intent":             final.intent,
            "compliance_passed":  final.compliance_passed,
            "caveats":            final.caveats,
            "faithfulness":       round(final.faithfulness, 3),
            "answer_relevancy":   round(final.answer_relevancy, 3),
            "context_precision":  round(final.context_precision, 3),
            "latency_ms":         round(final.latency_ms),
        }

    def print_result(self, result: dict):
        """Pretty-print the final HR Copilot response."""
        hdr("HR COPILOT RESPONSE", CG)
        print(f"\n  {CB}Q: {result['question']}{CR2}\n")

        for para in result["answer"].split("\n"):
            if para.strip():
                print(textwrap.fill(para.strip(), 72, initial_indent="  ", subsequent_indent="    "))

        print(f"\n  {CY}Sources:{CR2}  {', '.join(result['sources']) or 'none'}")
        print(f"  {CY}Agents:{CR2}   {', '.join(result['agents'])}")
        print(f"  {CY}Intent:{CR2}   {result['intent']}")
        if result["caveats"]:
            print(f"  {CY2}Caveats:{CR2}  {result['caveats']}")

        faith  = result["faithfulness"]
        rel    = result["answer_relevancy"]
        prec   = result["context_precision"]
        latency= result["latency_ms"]
        print(f"\n  {CY}Quality:{CR2}")
        print(f"    Faithfulness:      {faith:.2f}  {'✅' if faith>=FAITHFULNESS_GATE else '❌'}")
        print(f"    Answer Relevancy:  {rel:.2f}  {'✅' if rel>=0.68 else '⚠️'}")
        print(f"    Context Precision: {prec:.2f}")
        print(f"    Latency:           {latency} ms")
        print(f"    Compliance:        {'✅ PASS' if result['compliance_passed'] else '❌ BLOCKED'}")

    def run_eval_suite(self, gate: bool = False) -> bool:
        """Run evaluation on a predefined suite of HR questions."""
        questions = [
            # Leave
            "What is the maximum annual leave carry-forward and can it be encashed?",
            # Compensation
            "What is the salary band range for a Band 3 Senior Lead?",
            # Remote work
            "Can I work from home 3 days per week if I completed probation?",
            # Onboarding
            "What must I complete in my first week as a new employee?",
            # Data
            "Which department has the highest attrition and what is the total company headcount?",
            # Multi-domain
            "I want to understand my L&D budget and remote work options as a Band 2 employee.",
        ]

        hdr(f"EVALUATION SUITE — {len(questions)} HR Scenarios", CY)
        results = []
        for i, q in enumerate(questions, 1):
            print(f"\n  [{i}/{len(questions)}] {q}")
            r = self.ask(q, use_llm=False)
            self.print_result(r)
            results.append(r)

        avg_faith = sum(r["faithfulness"] for r in results) / len(results)
        avg_rel   = sum(r["answer_relevancy"] for r in results) / len(results)
        avg_lat   = sum(r["latency_ms"] for r in results) / len(results)
        passed    = avg_faith >= FAITHFULNESS_GATE

        hdr("EVALUATION SUMMARY", CG if passed else CR)
        print(f"  Questions:         {len(results)}")
        print(f"  Avg Faithfulness:  {avg_faith:.3f}  (gate: {FAITHFULNESS_GATE})")
        print(f"  Avg Relevancy:     {avg_rel:.3f}")
        print(f"  Avg Latency:       {avg_lat:.0f} ms")
        print(f"  CI/CD Gate:        {'✅ PASS' if passed else '❌ BLOCKED — deployment prevented'}")

        os.makedirs("data/eval", exist_ok=True)
        with open("data/eval/eval_suite_report.json","w") as f:
            json.dump({
                "gate_passed": passed,
                "avg_faithfulness": round(avg_faith,3),
                "avg_relevancy": round(avg_rel,3),
                "faithfulness_gate": FAITHFULNESS_GATE,
                "results": results,
            }, f, indent=2)
        ok("Report saved: data/eval/eval_suite_report.json")

        if gate:
            sys.exit(0 if passed else 1)
        return passed

    def interactive(self):
        """Interactive Q&A terminal."""
        hdr("HR COPILOT — Interactive Mode (Ctrl+C to exit)", CG)
        print("  Ask any HR question. Try:")
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
            if not q or q.lower() in ("exit","quit","q"):
                break
            result = self.ask(q, use_llm=True)
            self.print_result(result)


def main():
    parser = argparse.ArgumentParser(description="HR Copilot — Multi-Agent Pipeline")
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--eval",        action="store_true")
    parser.add_argument("--gate",        action="store_true")
    parser.add_argument("--question",    type=str, default=None)
    parser.add_argument("--quiet",       action="store_true")
    args = parser.parse_args()

    print(f"\n{CB}{CY}")
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║  HR Copilot — Multi-Agent Employee Self-Service     ║")
    print("  ║  Local Build Edition · MLDS 2026                    ║")
    print(f"  ╚══════════════════════════════════════════════════════╝{CR2}")

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

    # Default: demo queries then interactive
    pipeline._ensure_loaded()
    demos = [
        "What is the carry-forward limit for annual leave?",
        "What salary band range does a Band 4 manager fall in?",
        "I am joining next week — what documents do I need?",
    ]
    hdr("DEMO QUERIES", CY)
    for q in demos:
        result = pipeline.ask(q, use_llm=False)
        pipeline.print_result(result)

    pipeline.interactive()


if __name__ == "__main__":
    main()
