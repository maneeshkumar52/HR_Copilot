"""
============================================================
COMPONENT B — OrchestratorAgent (Multi-Agent Router)
============================================================
HR Copilot · MLDS 2026 · Maneesh Kumar & Ravikiran Ravada

Use Case:
  An employee asks: "Can I work from home 4 days a week and
  what will my carry-forward balance look like after the April
  increment?"
  
  This single question spans 3 HR domains (remote work, leave,
  compensation). A single monolithic agent answers poorly.
  The OrchestratorAgent decomposes it and routes each part to
  the right specialist agent.

What this builds:
  1. HR intent classification (9 HR-specific intents)
  2. Multi-agent routing decision (which agents to invoke)
  3. Query decomposition (independent sub-questions)
  4. HRQueryPlan dataclass — typed handoff to all downstream agents
  5. Ollama LLM planner + rule-based fallback

Azure equivalent:
  Azure OpenAI with structured output (response_format=JSON) for
  deterministic intent classification. Azure Service Bus for
  agent-to-agent message passing.

Run: python3 component_b_orchestrator_agent.py
Requires: Component A (data/index/) completed first.
============================================================
"""
import re, json
from typing import List, Optional
from hr_data_models import (
    QueryIntent, AgentName, HRQueryPlan
)

OLLAMA_MODEL = "mistral"
USE_OLLAMA   = True

print("="*60)
print("  COMPONENT B — OrchestratorAgent")
print("="*60)


# ─────────────────────────────────────────────────────────────
# ROUTING TABLE
# Maps intent → specialist agents to invoke
# This is the core multi-agent routing logic.
# ─────────────────────────────────────────────────────────────
ROUTING_TABLE: dict = {
    QueryIntent.LEAVE_POLICY:   [AgentName.POLICY_RAG, AgentName.COMPLIANCE],
    QueryIntent.COMPENSATION:   [AgentName.POLICY_RAG, AgentName.DATA_QUERY, AgentName.COMPLIANCE],
    QueryIntent.REMOTE_WORK:    [AgentName.POLICY_RAG, AgentName.COMPLIANCE],
    QueryIntent.ONBOARDING:     [AgentName.ONBOARDING, AgentName.COMPLIANCE],
    QueryIntent.GRIEVANCE:      [AgentName.POLICY_RAG, AgentName.COMPLIANCE],
    QueryIntent.LEARNING:       [AgentName.POLICY_RAG],
    QueryIntent.HEADCOUNT_DATA: [AgentName.DATA_QUERY],
    QueryIntent.SALARY_BAND:    [AgentName.DATA_QUERY, AgentName.COMPLIANCE],
    QueryIntent.MULTI_DOMAIN:   [AgentName.POLICY_RAG, AgentName.DATA_QUERY,
                                  AgentName.ONBOARDING, AgentName.COMPLIANCE],
    QueryIntent.UNKNOWN:        [AgentName.POLICY_RAG, AgentName.COMPLIANCE],
}

# Priority HR docs per intent (guides retrieval scope)
# Values must match the 'category' field in hr_chunks.json (set by HR_CATEGORY_MAP in component_a)
PRIORITY_DOCS_MAP: dict = {
    QueryIntent.LEAVE_POLICY:   ["leave"],
    QueryIntent.COMPENSATION:   ["compensation"],
    QueryIntent.REMOTE_WORK:    ["remote_work"],
    QueryIntent.ONBOARDING:     ["onboarding"],
    QueryIntent.GRIEVANCE:      ["grievance"],
    QueryIntent.LEARNING:       ["learning"],
    QueryIntent.HEADCOUNT_DATA: [],
    QueryIntent.SALARY_BAND:    [],
    QueryIntent.MULTI_DOMAIN:   [],
    QueryIntent.UNKNOWN:        [],
}

NEEDS_STRUCTURED_DATA = {
    QueryIntent.HEADCOUNT_DATA, QueryIntent.SALARY_BAND,
    QueryIntent.COMPENSATION, QueryIntent.MULTI_DOMAIN
}


# ─────────────────────────────────────────────────────────────
# RULE-BASED CLASSIFIER (instant, no LLM cost)
# ─────────────────────────────────────────────────────────────
INTENT_KEYWORDS = {
    QueryIntent.LEAVE_POLICY:   r"\b(leave|annual leave|sick leave|casual|carry.forward|maternity|paternity|bereavement|holiday|time off|pto|encash)\b",
    QueryIntent.COMPENSATION:   r"\b(salary|ctc|increment|hike|raise|bonus|allowance|benefit|pay|stipend|variable|performance.pay|appraisal|pf|provident|gratuity|esop)\b",
    QueryIntent.REMOTE_WORK:    r"\b(work from home|wfh|remote|hybrid|flexible|office|wfa|work from abroad|core hours|in.office)\b",
    QueryIntent.ONBOARDING:     r"\b(onboard|joining|new employee|day 1|first day|induction|welcome kit|offer letter|access card|laptop|buddy)\b",
    QueryIntent.GRIEVANCE:      r"\b(grievance|complaint|posh|harassment|disciplin|warning|pip|performance improvement|termination|notice|code of conduct|conflict)\b",
    QueryIntent.LEARNING:       r"\b(training|learning|l&d|l\s+and\s+d|lnd|certification|course|workshop|upskill|reimburs|udemy|coursera|conference|hackathon|mentoring|budget|clawback|sponsorship)\b",
    QueryIntent.HEADCOUNT_DATA: r"\b(headcount|attrition|how many employees|strength|team size|open positions|vacancy|turnover rate)\b",
    QueryIntent.SALARY_BAND:    r"\b(salary band|pay band|band [b1-6]|grade|level|salary range|what do .* earn|compensation range)\b",
}

def classify_intent_rules(question: str) -> QueryIntent:
    q = question.lower()
    scores = {}
    for intent, pattern in INTENT_KEYWORDS.items():
        matches = len(re.findall(pattern, q))
        if matches:
            scores[intent] = matches
    if not scores:
        return QueryIntent.UNKNOWN
    if len(scores) >= 3:
        return QueryIntent.MULTI_DOMAIN
    return max(scores, key=scores.get)


# ─────────────────────────────────────────────────────────────
# RULE-BASED DECOMPOSER
# ─────────────────────────────────────────────────────────────
def decompose_query_rules(question: str, intent: QueryIntent) -> List[str]:
    """
    Breaks a complex HR question into independently answerable sub-questions.
    Uses coordination detection (and/or/also) + intent-specific patterns.
    """
    q = question.strip()

    # Split on coordination conjunction patterns
    for connector in [" and also ", " as well as ", " additionally, ", " plus, "]:
        if connector in q.lower():
            parts = re.split(connector, q, flags=re.IGNORECASE, maxsplit=1)
            if len(parts) == 2 and len(parts[0]) > 15 and len(parts[1]) > 15:
                return [parts[0].strip(), parts[1].strip()]

    # Conditional multi-hop: "If I [X], what happens to [Y]?"
    conditional = re.match(r"if\s+(.+?),\s*(.+)", q, re.IGNORECASE)
    if conditional:
        condition  = conditional.group(1).strip()
        consequence= conditional.group(2).strip()
        return [
            f"What is the rule for: {condition}?",
            f"What is the outcome for: {consequence}?",
        ]

    # Multi-domain: smartly decompose into single-domain sub-queries
    if intent == QueryIntent.MULTI_DOMAIN:
        sub_qs = []
        
        # Extract ESOP/Band questions
        if re.search(r"\b(esop|stock)\b", q, re.IGNORECASE):
            band_match = re.search(r"band\s+(\d|[b\d]+)", q, re.IGNORECASE)
            if band_match:
                sub_qs.append(f"What is the ESOP policy for {band_match.group(1)}?")
            else:
                sub_qs.append("What is the ESOP eligibility and vesting schedule?")
        
        # Extract leave/carry-forward questions
        if re.search(r"\bleave\b|\bcarry.forward\b|\bencash\b", q, re.IGNORECASE):
            sub_qs.append("What is the annual leave carry forward limit?")
        
        # Extract notice period questions
        if re.search(r"\bnotice\b", q, re.IGNORECASE):
            band_match = re.search(r"band\s+(\d|[b\d]+)", q, re.IGNORECASE)
            if band_match:
                sub_qs.append(f"What is the notice period for {band_match.group(1)}?")
            else:
                sub_qs.append("What is the notice period for my band?")
        
        # Extract salary band questions
        if re.search(r"\bsalary.band\b|\bctc\b|\bpay\b|\bcompensation\b", q, re.IGNORECASE):
            band_match = re.search(r"band\s+(\d|[b\d]+)", q, re.IGNORECASE)
            if band_match:
                sub_qs.append(f"What is the salary band range for {band_match.group(1)}?")
        
        # If we extracted sub-questions, use them; otherwise use generic decomposition
        if sub_qs:
            return sub_qs[:3]  # Max 3 sub-queries per original question
        
        return [
            f"What is the HR policy for: {q}?",
            f"Are there any compliance requirements for: {q}?",
        ]

    return [q]


def get_priority_docs_for_subquery(sub_question: str, main_intent: QueryIntent) -> List[str]:
    """
    Maps a sub-question to specific HR document categories for precise retrieval.
    This prevents MULTI_DOMAIN questions from retrieving from all docs.
    
    Returns: List of priority_docs keys (match HR_CATEGORY_MAP values)
    """
    q = sub_question.lower()
    
    # ESOP → compensation
    if "esop" in q or "stock" in q:
        return ["compensation"]
    
    # Leave → leave_policy
    if "leave" in q or "carry" in q or "encash" in q:
        return ["leave"]
    
    # Notice period → compensation
    if "notice" in q:
        return ["compensation"]
    
    # Salary band → compensation (will force JSON lookup)
    if "salary" in q or "band" in q or "ctc" in q or "compensation" in q:
        return ["compensation"]
    
    # Remote work → remote_work
    if "remote" in q or "wfh" in q or "work from home" in q:
        return ["remote_work"]
    
    # Onboarding → onboarding
    if "joining" in q or "onboard" in q or "day 1" in q or "first week" in q:
        return ["onboarding"]
    
    # Grievance → grievance
    if "grievance" in q or "complaint" in q or "posh" in q:
        return ["grievance"]
    
    # Learning → learning
    if "training" in q or "l&d" in q or "learning" in q or "budget" in q:
        return ["learning"]
    
    # No specific match → use empty list (retrieve from all)
    return []


# ─────────────────────────────────────────────────────────────
# LLM-BASED ORCHESTRATOR (Ollama/Mistral)
# ─────────────────────────────────────────────────────────────
ORCHESTRATOR_PROMPT = """You are the OrchestratorAgent for an enterprise HR Copilot system.
You must analyse an employee's HR question and produce a routing plan.

INTENT OPTIONS:
leave_policy, compensation, remote_work, onboarding, grievance, learning,
headcount_data, salary_band, multi_domain, unknown

AGENT OPTIONS:
PolicyRAGAgent     — answers policy questions from HR documents
DataQueryAgent     — queries structured data (headcount CSV, salary bands JSON)
OnboardingAgent    — new joiner specific questions and checklists
ComplianceGuardAgent — validates compliance/legal accuracy

RULES:
- If question spans 2+ domains → intent=multi_domain, invoke all relevant agents
- If question asks for numbers/counts/statistics → include DataQueryAgent
- Always include ComplianceGuardAgent for sensitive topics (POSH, disciplinary, termination)
- Decompose complex questions into 2-3 INDEPENDENT sub-questions max

Respond ONLY with this JSON format (no markdown, no explanation):
{
  "intent": "<intent>",
  "agents": ["<agent1>", "<agent2>"],
  "sub_queries": ["<q1>", "<q2>"],
  "priority_docs": ["<doc1>"],
  "needs_structured_data": false,
  "reasoning": "<one sentence>",
  "confidence": 0.9
}"""

_INTENT_STR_MAP = {
    "leave_policy":    QueryIntent.LEAVE_POLICY,
    "compensation":    QueryIntent.COMPENSATION,
    "remote_work":     QueryIntent.REMOTE_WORK,
    "onboarding":      QueryIntent.ONBOARDING,
    "grievance":       QueryIntent.GRIEVANCE,
    "learning":        QueryIntent.LEARNING,
    "headcount_data":  QueryIntent.HEADCOUNT_DATA,
    "salary_band":     QueryIntent.SALARY_BAND,
    "multi_domain":    QueryIntent.MULTI_DOMAIN,
    "unknown":         QueryIntent.UNKNOWN,
}
_AGENT_STR_MAP = {
    "PolicyRAGAgent":      AgentName.POLICY_RAG,
    "DataQueryAgent":      AgentName.DATA_QUERY,
    "OnboardingAgent":     AgentName.ONBOARDING,
    "ComplianceGuardAgent":AgentName.COMPLIANCE,
    "OrchestratorAgent":   AgentName.ORCHESTRATOR,
}

def orchestrate_with_llm(question: str) -> Optional[HRQueryPlan]:
    try:
        import ollama
        r = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role":"system","content": ORCHESTRATOR_PROMPT},
                {"role":"user",  "content": f"HR Question: {question}"},
            ],
            options={"temperature": 0.0},
        )
        raw = r["message"]["content"]
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            return None
        d = json.loads(m.group())

        intent  = _INTENT_STR_MAP.get(d.get("intent",""), QueryIntent.UNKNOWN)
        agents  = [_AGENT_STR_MAP[a] for a in d.get("agents",[]) if a in _AGENT_STR_MAP]
        if not agents:
            agents = ROUTING_TABLE[intent]

        return HRQueryPlan(
            original_question = question,
            intent            = intent,
            sub_queries       = d.get("sub_queries", [question]),
            agents_to_invoke  = agents,
            priority_docs     = d.get("priority_docs", PRIORITY_DOCS_MAP.get(intent,[])),
            needs_structured  = d.get("needs_structured_data", intent in NEEDS_STRUCTURED_DATA),
            reasoning         = d.get("reasoning", ""),
            confidence        = float(d.get("confidence", 0.8)),
        )
    except Exception as e:
        print(f"  [Orchestrator] LLM error: {e} → rule-based fallback")
        return None


def orchestrate_rules(question: str) -> HRQueryPlan:
    intent  = classify_intent_rules(question)
    agents  = ROUTING_TABLE.get(intent, [AgentName.POLICY_RAG])
    sub_qs  = decompose_query_rules(question, intent)
    p_docs  = PRIORITY_DOCS_MAP.get(intent, [])
    return HRQueryPlan(
        original_question = question,
        intent            = intent,
        sub_queries       = sub_qs,
        agents_to_invoke  = agents,
        priority_docs     = p_docs,
        needs_structured  = (intent in NEEDS_STRUCTURED_DATA),
        reasoning         = f"Rule-based: matched '{intent.value}' patterns",
        confidence        = 0.72,
    )


def orchestrator_agent(question: str, use_llm: bool = USE_OLLAMA) -> HRQueryPlan:
    """
    Main entry point for Component B.
    Returns an HRQueryPlan that tells all downstream agents:
    - What the question intends
    - Which agents should handle it
    - How to decompose it into sub-queries
    - Which HR documents to prioritise

    MULTI-AGENT ROUTING PRINCIPLE:
    Different HR questions need different expertise. Routing to specialist
    agents avoids the "one model answers everything poorly" problem.
    Each agent is prompted specifically for its domain → higher accuracy.
    """
    print(f"\n  [OrchestratorAgent] Routing: '{question[:70]}...'")
    plan = None
    if use_llm:
        plan = orchestrate_with_llm(question)

    if plan is None:
        plan = orchestrate_rules(question)

    print(f"  [OrchestratorAgent] Intent: {plan.intent.value}")
    print(f"  [OrchestratorAgent] Agents: {[a.value for a in plan.agents_to_invoke]}")
    print(f"  [OrchestratorAgent] Sub-queries: {len(plan.sub_queries)}")
    for i, q in enumerate(plan.sub_queries, 1):
        print(f"    {i}. {q}")
    return plan


# ─────────────────────────────────────────────────────────────
# MAIN — test routing for 6 different HR scenarios
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    scenarios = [
        ("What is the maximum annual leave I can carry forward to next year?",    "leave_policy"),
        ("I have a POSH complaint — what is the process?",                         "grievance"),
        ("Can I work from home full week? I joined 3 months ago.",                  "remote_work"),
        ("What documents do I need to submit on my first day?",                     "onboarding"),
        ("What is the salary band range for a Band 4 manager?",                     "salary_band"),
        ("If I take 5 days of casual leave and then need sick leave, what is my broadband allowance and increment timeline?", "multi_domain"),
    ]

    print(f"\n  Testing OrchestratorAgent routing on {len(scenarios)} HR scenarios\n")
    for question, expected in scenarios:
        plan = orchestrator_agent(question, use_llm=False)
        agents = [a.value for a in plan.agents_to_invoke]
        ok = "✅" if plan.intent.value == expected else f"⚠️ (expected {expected})"
        print(f"  {ok}  Agents: {agents}\n")

    print("="*60)
    print("  COMPONENT B COMPLETE ✅")
    print("  Run next: python3 component_c_policy_data_agents.py")
    print("="*60)
