"""
============================================================
HR COPILOT — Shared Data Models
All agents share these typed dataclasses as contracts.
============================================================
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class QueryIntent(Enum):
    """Intent categories specific to HR domain queries."""
    LEAVE_POLICY       = "leave_policy"        # Leave entitlements, carry-forward
    COMPENSATION       = "compensation"         # Salary, benefits, increments
    REMOTE_WORK        = "remote_work"          # WFH rules, flexible hours
    ONBOARDING         = "onboarding"           # New joiner process, checklists
    GRIEVANCE          = "grievance"            # Complaints, disciplinary, POSH
    LEARNING           = "learning"             # L&D budget, certifications
    HEADCOUNT_DATA     = "headcount_data"       # Structured data: headcount, attrition
    SALARY_BAND        = "salary_band"          # Structured data: salary ranges
    MULTI_DOMAIN       = "multi_domain"         # Spans multiple HR domains
    UNKNOWN            = "unknown"              # Cannot classify


class AgentName(Enum):
    ORCHESTRATOR   = "OrchestratorAgent"
    POLICY_RAG     = "PolicyRAGAgent"
    DATA_QUERY     = "DataQueryAgent"
    ONBOARDING     = "OnboardingAgent"
    COMPLIANCE     = "ComplianceGuardAgent"


@dataclass
class HRQueryPlan:
    """
    Typed contract produced by the OrchestratorAgent.
    Downstream agents read this — never pass raw strings between agents.
    """
    original_question:  str
    intent:             QueryIntent         = QueryIntent.UNKNOWN
    sub_queries:        List[str]           = field(default_factory=list)
    agents_to_invoke:   List[AgentName]     = field(default_factory=list)
    priority_docs:      List[str]           = field(default_factory=list)
    # e.g. ["leave_policy", "compensation_benefits"]
    needs_structured:   bool                = False
    # True if query needs CSV/JSON data (headcount, salary bands)
    reasoning:          str                 = ""
    confidence:         float               = 0.80

    def summary(self) -> str:
        agents = [a.value for a in self.agents_to_invoke]
        return (
            f"  Intent:   {self.intent.value}\n"
            f"  Agents:   {agents}\n"
            f"  Sub-Qs:   {len(self.sub_queries)}\n"
            + "\n".join(f"    - {q}" for q in self.sub_queries)
            + f"\n  Docs:     {self.priority_docs}"
            + f"\n  Struct:   {self.needs_structured}"
        )


@dataclass
class RetrievedChunk:
    """A single retrieved knowledge chunk."""
    chunk_id:      str
    text:          str
    source_file:   str
    category:      str           # which HR domain
    rrf_score:     float = 0.0
    rerank_score:  float = 0.0
    entail_score:  float = 0.0
    verified:      bool  = False
    agent_source:  str   = ""    # which agent retrieved this


@dataclass
class AgentResponse:
    """
    The output of a single specialist agent.
    OrchestratorAgent merges multiple AgentResponses.
    """
    agent:         AgentName
    answer:        str
    sources:       List[str]     = field(default_factory=list)
    confidence:    float         = 0.0
    chunks_used:   int           = 0
    data_used:     bool          = False   # True if structured data was used
    warnings:      List[str]     = field(default_factory=list)


@dataclass
class ComplianceCheckResult:
    """Result from ComplianceGuardAgent."""
    passes:        bool
    confidence:    float
    flags:         List[str]     = field(default_factory=list)
    # flags = list of compliance issues found
    corrected_answer: Optional[str] = None
    # If passes=False, corrected_answer has the safe version


@dataclass
class FinalHRResponse:
    """
    The complete, merged, compliance-checked response delivered to the employee.
    This is what the user sees.
    """
    question:             str
    answer:               str
    sources:              List[str]
    agents_contributed:   List[str]
    intent:               str
    compliance_passed:    bool
    faithfulness:         float = 0.0
    answer_relevancy:     float = 0.0
    context_precision:    float = 0.0
    latency_ms:           float = 0.0
    caveats:              List[str] = field(default_factory=list)
