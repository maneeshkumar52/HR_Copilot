"""
============================================================
MULTI-AGENT FRAMEWORK — Core Infrastructure
============================================================
HR Copilot · Maneesh Kumar

This module is the backbone of the multi-agent system.
It defines HOW agents are built, registered, discovered,
and executed — independently of any HR-specific logic.

────────────────────────────────────────────────────────────
WHAT IS A MULTI-AGENT SYSTEM?
────────────────────────────────────────────────────────────
A Multi-Agent System (MAS) is a collection of autonomous
software agents that each specialise in a narrow task and
collaborate to solve problems too complex for any single
agent.

Why use it here?
  HR covers 9+ completely different domains: leave policy,
  compensation, POSH compliance, onboarding, analytics…
  One monolithic LLM performs poorly across all of them.
  Specialist agents (one per domain) outperform a single
  general-purpose agent at the cost of coordination logic.

────────────────────────────────────────────────────────────
COMPONENTS IN THIS FILE
────────────────────────────────────────────────────────────
  BaseAgent            — Abstract interface every agent implements
  PlannerAgent         — Interface for agents that route queries
  AgentRegistry        — Central registry (service discovery)
  AgentMessage         — Typed envelope for inter-agent comms
  AgentTask            — Execution state tracker per agent
  AgentStatus          — Enum: IDLE / RUNNING / COMPLETED / FAILED
  ParallelAgentExecutor — Runs specialist agents concurrently

────────────────────────────────────────────────────────────
AZURE PRODUCTION EQUIVALENTS
────────────────────────────────────────────────────────────
  BaseAgent            → Azure Container App (one container per agent)
  AgentRegistry        → Azure Service Discovery / API Management
  AgentMessage         → Azure Service Bus message
  ParallelAgentExecutor → Azure Durable Functions fan-out pattern
============================================================
"""

import time
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable

from hr_data_models import AgentName, HRQueryPlan, AgentResponse


print("="*60)
print("  MULTI-AGENT FRAMEWORK — agent_framework.py loaded")
print("="*60)


# ─────────────────────────────────────────────────────────────
# AGENT STATUS ENUM
# Tracks the lifecycle of each agent during a pipeline run.
# ─────────────────────────────────────────────────────────────
class AgentStatus(Enum):
    """
    Lifecycle states of an agent.

    IDLE         → Registered but not yet called
    RUNNING      → Currently processing a query (in a thread)
    COMPLETED    → Finished successfully; result is available
    FAILED       → Threw an exception; error field has details
    UNREGISTERED → Not in the registry (looked up but not found)
    """
    IDLE         = "idle"
    RUNNING      = "running"
    COMPLETED    = "completed"
    FAILED       = "failed"
    UNREGISTERED = "unregistered"


# ─────────────────────────────────────────────────────────────
# AGENT MESSAGE
# The typed envelope for agent-to-agent communication.
#
# KEY CONCEPT — Loose Coupling via Messages:
#   Agents do NOT call each other directly.
#   They send typed AgentMessage objects.
#   This means:
#     1. Agents can be replaced without changing callers.
#     2. Every message can be logged for observability.
#     3. Messages can be queued for async / retry.
#
# Azure equivalent: Azure Service Bus message with JSON body.
# ─────────────────────────────────────────────────────────────
@dataclass
class AgentMessage:
    """
    Typed message passed between agents.

    Fields:
        sender    — who created this message (AgentName enum)
        recipient — intended receiver (None = broadcast to all)
        payload   — the actual data: HRQueryPlan (request) or
                    AgentResponse (reply)
        msg_type  — "request" | "response" | "error"
        timestamp — Unix epoch when the message was created
        trace_id  — optional ID for distributed tracing
    """
    sender:    AgentName
    recipient: Optional[AgentName]
    payload:   Any                     # HRQueryPlan or AgentResponse
    msg_type:  str  = "request"        # "request" | "response" | "error"
    timestamp: float = field(default_factory=time.time)
    trace_id:  str  = ""


# ─────────────────────────────────────────────────────────────
# AGENT TASK
# Captures the execution state of ONE agent in a parallel run.
# ─────────────────────────────────────────────────────────────
@dataclass
class AgentTask:
    """
    Execution trace for a single agent invocation.

    WHY TRACK TASKS?
        When 3 agents run in parallel we need to know:
          - Which ones finished, which are still running?
          - How long did each take? (bottleneck detection)
          - Did any fail? (graceful degradation)
        AgentTask answers all three questions.

    The ParallelAgentExecutor creates one AgentTask per agent.
    """
    agent_name: AgentName
    status:     AgentStatus             = AgentStatus.IDLE
    result:     Optional[AgentResponse] = None
    error:      Optional[str]           = None
    start_time: float                   = 0.0
    end_time:   float                   = 0.0

    @property
    def latency_ms(self) -> float:
        """Wall-clock execution time in milliseconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    def __repr__(self) -> str:
        return (
            f"AgentTask({self.agent_name.value} | "
            f"{self.status.value} | {self.latency_ms:.0f}ms)"
        )


# ─────────────────────────────────────────────────────────────
# BASE AGENT
# Every specialist agent MUST extend this class.
#
# KEY CONCEPT — Uniform Interface (Liskov Substitution Principle):
#   The pipeline calls agent.run(plan) on every specialist agent.
#   It doesn't need to know whether it's talking to a RAG agent,
#   a database agent, or a checklist agent.
#   All that matters is: give it a plan, get back a response.
#
#   This is exactly how Azure Container Apps work in production:
#   each agent is an HTTP service with the same POST /run endpoint.
# ─────────────────────────────────────────────────────────────
class BaseAgent(ABC):
    """
    Abstract base class for all HR Copilot specialist agents.

    CONTRACT (every subclass MUST implement):
        agent_name → property: return AgentName enum value
        run(plan)  → method: accept HRQueryPlan, return AgentResponse

    OPTIONAL overrides:
        initialize() → load models, open DB connections
        shutdown()   → release resources

    SUBCLASSES:
        PolicyRAGAgent    (component_c) — hybrid policy retrieval
        DataQueryAgent    (component_c) — structured data lookup
        OnboardingAgent   (component_d) — new joiner checklist

    NOT subclasses (different interface):
        OrchestratorAgent  — produces HRQueryPlan (not AgentResponse)
        ComplianceGuard    — takes (question, chunks), not just plan
        ResponseSynthesizer — merges all agents' responses
    """

    @property
    @abstractmethod
    def agent_name(self) -> AgentName:
        """Return the unique AgentName enum for this agent."""
        ...

    @abstractmethod
    def run(self, plan: HRQueryPlan) -> AgentResponse:
        """
        Process an HR query plan and return a structured response.

        Args:
            plan: HRQueryPlan produced by OrchestratorAgent.
                  Contains intent, sub_queries, priority_docs, etc.

        Returns:
            AgentResponse with answer text, sources, confidence score.
        """
        ...

    def initialize(self) -> None:
        """
        One-time setup: load ML models, connect to databases.
        Called by AgentRegistry.register() on first registration.
        Override in subclasses that need pre-loading.
        """
        pass

    def shutdown(self) -> None:
        """
        Graceful shutdown: release GPU memory, close connections.
        Called by AgentRegistry.shutdown_all() when pipeline exits.
        Override in subclasses that hold open resources.
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}[{self.agent_name.value}]>"


# ─────────────────────────────────────────────────────────────
# PLANNER AGENT
# A separate base for the OrchestratorAgent.
# Its job is different: it takes a raw question and produces
# a routing plan — it does NOT retrieve or answer.
# ─────────────────────────────────────────────────────────────
class PlannerAgent(ABC):
    """
    Base class for agents that PLAN (not answer).

    The OrchestratorAgent extends this.
    Input:  raw employee question (string)
    Output: HRQueryPlan (intent, agents to invoke, sub-queries)

    WHY SEPARATE FROM BaseAgent?
        BaseAgent: question → answer
        PlannerAgent: question → routing plan
        These are fundamentally different responsibilities.
        Mixing them would violate Single Responsibility Principle.
    """

    @abstractmethod
    def plan(self, question: str, use_llm: bool = True) -> HRQueryPlan:
        """
        Analyse a question and produce a routing plan.

        Args:
            question: Raw employee HR question.
            use_llm:  Whether to use LLM for classification.
        Returns:
            HRQueryPlan describing how to route and decompose the query.
        """
        ...


# ─────────────────────────────────────────────────────────────
# AGENT REGISTRY
# Singleton: knows about every agent in the system.
#
# KEY CONCEPT — Service Discovery:
#   The pipeline asks "who handles POLICY_RAG?" not "where is
#   the PolicyRAGAgent class?".
#   This lets you swap implementations without touching the pipeline.
#
#   Azure equivalent: Azure API Management as a service gateway,
#   or Kubernetes Service Discovery.
# ─────────────────────────────────────────────────────────────
class AgentRegistry:
    """
    Central registry for all HR Copilot agents.

    Pattern: Service Locator
    The OrchestratorAgent decides WHICH agents to call.
    The Registry tells the pipeline WHERE they are.
    New agents can be added without touching the pipeline.

    Usage:
        registry = AgentRegistry()
        registry.register(PolicyRAGAgent(...))
        registry.register(DataQueryAgent())

        agent = registry.get(AgentName.POLICY_RAG)
        response = agent.run(plan)

        print(registry)  # <AgentRegistry agents=[PolicyRAGAgent, ...]>
    """

    def __init__(self):
        self._agents: Dict[AgentName, BaseAgent] = {}
        self._statuses: Dict[AgentName, AgentStatus] = {}
        self._lock = threading.Lock()      # Thread-safe for parallel access

    def register(self, agent: BaseAgent) -> None:
        """
        Register an agent. Calls agent.initialize() once.
        Thread-safe: can be called from multiple threads.
        """
        with self._lock:
            if not isinstance(agent, BaseAgent):
                raise TypeError(
                    f"{agent!r} must extend BaseAgent to be registered. "
                    f"Got {type(agent).__name__} instead."
                )
            name = agent.agent_name
            if name not in self._agents:
                agent.initialize()          # One-time setup
            self._agents[name] = agent
            self._statuses[name] = AgentStatus.IDLE
            print(f"  [Registry] ✅ Registered: {agent}")

    def get(self, name: AgentName) -> Optional[BaseAgent]:
        """Look up a registered agent by its AgentName enum."""
        return self._agents.get(name)

    def list_agents(self) -> List[AgentName]:
        """Return names of all registered agents."""
        return list(self._agents.keys())

    def get_status(self, name: AgentName) -> AgentStatus:
        """Return the current status of a named agent."""
        return self._statuses.get(name, AgentStatus.UNREGISTERED)

    def set_status(self, name: AgentName, status: AgentStatus) -> None:
        """Update agent status (called by ParallelAgentExecutor)."""
        with self._lock:
            self._statuses[name] = status

    def shutdown_all(self) -> None:
        """Gracefully shut down all registered agents."""
        for agent in self._agents.values():
            try:
                agent.shutdown()
            except Exception:
                pass
        self._agents.clear()
        self._statuses.clear()
        print("  [Registry] All agents shut down.")

    def __len__(self) -> int:
        return len(self._agents)

    def __repr__(self) -> str:
        names = [a.value for a in self._agents.keys()]
        return f"<AgentRegistry agents={names}>"


# ─────────────────────────────────────────────────────────────
# PARALLEL AGENT EXECUTOR
# The KEY innovation of this multi-agent framework.
#
# KEY CONCEPT — Fan-Out / Fan-In:
#   1. FAN-OUT: submit all specialist agents to a thread pool
#      simultaneously (no waiting for one to finish before
#      starting the next).
#   2. FAN-IN: collect results as they arrive (as_completed)
#      and merge them in the pipeline.
#
#   Sequential:  RAG(3s) → DataQuery(1s) → Onboarding(2s) = 6s
#   Parallel:    all three at once         → ~3s (bounded by slowest)
#
#   Azure equivalent: Azure Durable Functions fan-out pattern.
#   Each activity function maps to one specialist agent.
# ─────────────────────────────────────────────────────────────
class ParallelAgentExecutor:
    """
    Executes specialist agents CONCURRENTLY using a thread pool.

    Design:
        - Uses concurrent.futures.ThreadPoolExecutor for I/O-bound agents
        - Results collected with as_completed (not wait) for early returns
        - Each agent gets an isolated copy of the plan (no shared state)
        - AgentTask records timing + status for the pipeline trace

    Thread Safety:
        Each agent.run(plan) call is independent.
        Agents do NOT write to shared mutable state.
        Results collected and merged by the caller after execution.

    Limitations (local build):
        Threads share a process — GIL limits true CPU parallelism.
        For CPU-bound ML inference, production uses separate processes
        or Azure Container Apps (separate containers).

    Usage:
        executor = ParallelAgentExecutor(registry, max_workers=4)
        responses, tasks = executor.execute(plan, plan.agents_to_invoke)

        for task in tasks:
            print(task.agent_name.value, task.latency_ms)
    """

    # These agents are "specialist agents" that fit BaseAgent.run(plan).
    # ComplianceGuardAgent and OrchestratorAgent have different interfaces.
    SPECIALIST_AGENTS = {AgentName.POLICY_RAG, AgentName.DATA_QUERY, AgentName.ONBOARDING}

    def __init__(self, registry: AgentRegistry, max_workers: int = 4):
        self.registry    = registry
        self.max_workers = max_workers

    def execute(
        self,
        plan: HRQueryPlan,
        agent_names: List[AgentName],
        on_complete: Optional[Callable[["AgentTask"], None]] = None,
    ) -> tuple:
        """
        Run all requested specialist agents in parallel.

        Args:
            plan:         HRQueryPlan from OrchestratorAgent
            agent_names:  Which agents to invoke
                          (from plan.agents_to_invoke)
            on_complete:  Optional callback(AgentTask) called each
                          time an agent finishes — useful for streaming
                          results to a UI as they arrive.

        Returns:
            (responses, tasks)
            responses: List[AgentResponse] sorted by confidence desc
            tasks:     List[AgentTask] with timing + status per agent
        """
        to_run = [
            n for n in agent_names
            if n in self.SPECIALIST_AGENTS and self.registry.get(n) is not None
        ]

        if not to_run:
            return [], []

        tasks: Dict[AgentName, AgentTask] = {
            name: AgentTask(agent_name=name) for name in to_run
        }
        responses: List[AgentResponse] = []
        future_to_name: Dict[Future, AgentName] = {}

        print(
            f"\n  [ParallelExecutor] Launching {len(to_run)} agents "
            f"concurrently: {[n.value for n in to_run]}"
        )

        # ── STEP 1: Fan-out — submit all agents to thread pool ─────────────
        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(to_run)),
            thread_name_prefix="AgentWorker",
        ) as executor:

            for name in to_run:
                agent = self.registry.get(name)
                task  = tasks[name]
                task.status     = AgentStatus.RUNNING
                task.start_time = time.time()
                self.registry.set_status(name, AgentStatus.RUNNING)

                # Submit agent.run(plan) to thread pool
                # Each call is independent — no shared mutable state
                future = executor.submit(agent.run, plan)
                future_to_name[future] = name

            # ── STEP 2: Fan-in — collect results as they arrive ────────────
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                task = tasks[name]
                task.end_time = time.time()

                try:
                    result = future.result()
                    task.result = result
                    task.status = AgentStatus.COMPLETED
                    self.registry.set_status(name, AgentStatus.COMPLETED)
                    responses.append(result)
                    print(
                        f"  [ParallelExecutor] ✅ {name.value} "
                        f"completed in {task.latency_ms:.0f}ms"
                    )
                except Exception as exc:
                    task.status = AgentStatus.FAILED
                    task.error  = str(exc)
                    self.registry.set_status(name, AgentStatus.FAILED)
                    print(f"  [ParallelExecutor] ❌ {name.value} failed: {exc}")

                # Fire optional callback (e.g., stream partial result to UI)
                if on_complete:
                    on_complete(task)

        # Sort by confidence so highest-quality response is first
        responses.sort(key=lambda r: r.confidence, reverse=True)
        return responses, list(tasks.values())

    def print_execution_summary(self, tasks: List[AgentTask]) -> None:
        """Print a formatted timing summary of the parallel run."""
        if not tasks:
            return
        total = max((t.end_time for t in tasks if t.end_time), default=0) - \
                min((t.start_time for t in tasks if t.start_time), default=0)
        print(f"\n  ┌─ Parallel Execution Summary {'─'*28}┐")
        for t in sorted(tasks, key=lambda x: x.start_time):
            status_icon = "✅" if t.status == AgentStatus.COMPLETED else "❌"
            bar = "█" * max(1, int(t.latency_ms / 200))
            print(
                f"  │  {status_icon} {t.agent_name.value:20s} "
                f"{bar:<15s} {t.latency_ms:>6.0f}ms"
            )
        print(f"  │  {'Total wall-clock time:':35s} {total*1000:>6.0f}ms")
        print(f"  └{'─'*50}┘")


# ─────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  Testing multi-agent framework components...\n")

    # Test AgentMessage
    from hr_data_models import AgentName, QueryIntent, HRQueryPlan
    msg = AgentMessage(
        sender=AgentName.ORCHESTRATOR,
        recipient=AgentName.POLICY_RAG,
        payload="test payload",
        msg_type="request",
    )
    print(f"  AgentMessage: sender={msg.sender.value}, type={msg.msg_type}")

    # Test AgentTask
    task = AgentTask(agent_name=AgentName.POLICY_RAG)
    task.start_time = time.time() - 1.5
    task.end_time   = time.time()
    task.status     = AgentStatus.COMPLETED
    print(f"  AgentTask: {task}")

    # Test AgentRegistry with a concrete agent
    class _TestAgent(BaseAgent):
        @property
        def agent_name(self) -> AgentName:
            return AgentName.DATA_QUERY

        def run(self, plan: HRQueryPlan) -> None:
            return None

    registry = AgentRegistry()
    registry.register(_TestAgent())
    print(f"  Registry: {registry}")
    assert registry.get(AgentName.DATA_QUERY) is not None
    assert registry.get(AgentName.POLICY_RAG) is None

    print("\n  ✅ agent_framework.py — all components OK")
    print("="*60)
