"""
Dr. Hypothesis - Differential Diagnosis Agent.

Maintains and updates differential diagnosis with Bayesian probability reasoning.
"""

from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from langgraph.types import Command

from ..state import DiagnosticState, AgentRole
from ..tools import update_differential_diagnosis


def get_hypothesis_prompt(state: DiagnosticState) -> str:
    """Get the system prompt for the hypothesis agent with dynamic context."""
    
    dynamic_context = ""
    
    # Add situational context based on state
    if state.get("current_budget", 0) > 0 and state["enable_budget_tracking"]:
        remaining_budget = state["current_budget"] - state["cumulative_cost"]
        if remaining_budget < 1000:
            dynamic_context += f"""
**SITUATIONAL CONTEXT: BUDGET CONSTRAINT**
Remaining budget is low (${remaining_budget}). Focus on high-probability diagnoses that can be confirmed cost-effectively.
"""
    
    # Check for high confidence scenarios
    if state.get("differential_diagnosis"):
        max_confidence = max(item.probability for item in state["differential_diagnosis"])
        if max_confidence > 0.75:
            dynamic_context += f"""
**SITUATIONAL CONTEXT: HIGH CONFIDENCE**
Leading hypothesis has {max_confidence:.0%} confidence. Focus on confirming this diagnosis or identifying what single piece of evidence would push confidence >85%.
"""
    
    base_prompt = f"""{dynamic_context}

MANDATE: Keep an up-to-date, probability-ranked differential diagnosis.

ROLE: You are Dr. Hypothesis, the panel's expert in differential diagnosis and Bayesian reasoning.

RESPONSIBILITIES:
1. Maintain a probability-ranked differential diagnosis (2-5 most likely conditions)
2. Update probabilities using Bayesian reasoning after each new finding
3. Consider both common and rare diseases appropriate to the clinical context
4. Track how new evidence changes your diagnostic thinking

APPROACH:
1. Start with the most likely diagnoses based on presenting symptoms
2. For each new piece of evidence, consider:
   - How it supports or refutes each hypothesis
   - Whether it suggests new diagnoses to consider  
   - How it changes the relative probabilities
3. Always explain your Bayesian reasoning clearly

OUTPUT REQUIREMENTS:
- You MUST call update_differential_diagnosis() with structured output
- Provide 2-5 diagnoses with probabilities (0-1) and brief rationales
- List key supporting evidence for leading hypotheses
- Note any contradictory evidence that challenges current thinking

Focus on the most likely diagnoses and be concise in your rationales."""
    
    return base_prompt


def hypothesis_agent(state: DiagnosticState, model: BaseChatModel) -> Command:
    """
    Dr. Hypothesis agent for maintaining differential diagnosis.
    
    Args:
        state: Current diagnostic state
        model: Language model to use
        
    Returns:
        Command to update state and continue workflow
    """
    # Get the system prompt
    system_prompt = get_hypothesis_prompt(state)
    
    # Get the current case context
    case_context = f"""
**Current Case Information:**
{state['initial_vignette']}

**Previous Test Results:**
{_format_test_results(state.get('test_results', {}))}

**Available Clinical Findings:**
{_format_available_findings(state.get('available_findings', {}))}

**Current Iteration:** {state.get('iteration', 0)}
**Cumulative Cost:** ${state.get('cumulative_cost', 0)}
"""
    
    # Bind the update_differential_diagnosis tool
    model_with_tools = model.bind_tools([update_differential_diagnosis])
    
    # Create the input messages
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=case_context)
    ]
    
    # Add recent conversation context (temporarily disabled for debugging)
    # recent_messages = state["messages"][-3:] if len(state["messages"]) > 3 else state["messages"]
    # messages.extend(recent_messages)
    
    # Invoke the model
    response = model_with_tools.invoke(messages)
    
    # The tool will handle the state update and return a Command
    # If no tool was called, return to consensus
    if not response.tool_calls:
        return Command(
            goto="consensus",
            update={
                "messages": state["messages"] + [response],
                "hypothesis_analysis": "Analysis provided without structured differential diagnosis update",
            }
        )
    
    # Tool execution will be handled by the graph
    return Command(
        goto="consensus",
        update={"messages": state["messages"] + [response]}
    )


def _format_test_results(test_results: Dict[str, Any]) -> str:
    """Format test results for display."""
    if not test_results:
        return "No tests completed yet."
    
    formatted = []
    for test, result in test_results.items():
        if result.get("results") and result["results"] != "pending":
            formatted.append(f"- {test}: {result['results']}")
    
    return "\n".join(formatted) if formatted else "Tests ordered but results pending."


def _format_available_findings(findings: Dict[str, Any]) -> str:
    """Format available clinical findings."""
    if not findings:
        return "Request additional information through the Gatekeeper."
    
    formatted = []
    for category, details in findings.items():
        formatted.append(f"- {category}: {details}")
    
    return "\n".join(formatted)