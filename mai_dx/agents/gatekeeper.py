"""
Gatekeeper - Clinical Information Oracle.

Provides realistic clinical information disclosure based on the full case details.
"""

from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel 
from langgraph.types import Command

from ..state import DiagnosticState


def gatekeeper_agent(state: DiagnosticState, model: BaseChatModel) -> Command:
    """
    Gatekeeper agent for providing clinical information and test results.
    
    Args:
        state: Current diagnostic state
        model: Language model to use
        
    Returns:
        Command to continue diagnostic process
    """
    system_prompt = """
ROLE: You are the Gatekeeper, the clinical information oracle who provides realistic disclosure of patient information and test results.

RESPONSIBILITIES:
1. Answer specific clinical questions based on available case information
2. Provide test results for ordered tests
3. Simulate realistic information disclosure (some details may not be immediately apparent)
4. Maintain clinical realism in information timing and availability

APPROACH:
- Answer specific questions directly and accurately
- For test results: provide realistic findings based on the case
- Don't volunteer information not specifically requested
- Simulate real clinical scenarios where some information takes time to obtain

Be direct and factual in your responses.
"""
    
    # Get the current action/query
    current_action = state.get("current_action")
    
    # Handle both DiagnosticAction objects and dictionaries (due to serialization)
    if isinstance(current_action, dict):
        action_type = current_action.get("action_type")
        content = current_action.get("content")
    else:
        action_type = current_action.action_type if current_action else None
        content = current_action.content if current_action else None
        
    if current_action and action_type == "ask":
        query = content
    else:
        # Check if there's a test to provide results for
        pending_tests = [test for test, details in state.get("test_results", {}).items() 
                        if details.get("results") == "pending"]
        if pending_tests:
            query = f"Provide results for: {', '.join(pending_tests)}"
        else:
            query = "What additional clinical information is available?"
    
    # Prepare context
    context = f"""
**Full Case Details:**
{state['full_case_details']}

**Previously Disclosed Information:**
{_format_available_findings(state.get('available_findings', {}))}

**Tests Completed:**
{_format_completed_tests(state.get('test_results', {}))}

**Current Query:** {query}

Provide the requested information based on the full case details. Be clinically realistic about what would be available at this point in the workup.
"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context)
    ]
    
    # Invoke model
    response = model.invoke(messages)
    
    # Update state with new information
    new_findings = state.get("available_findings", {}).copy()
    
    # If this was about test results, mark tests as completed
    new_test_results = state.get("test_results", {}).copy()
    for test, details in new_test_results.items():
        if details.get("results") == "pending":
            new_test_results[test]["results"] = f"Results provided by Gatekeeper"
    
    # Update findings with new information (simplified)
    new_findings[f"Query_{state.get('iteration', 0)}"] = response.content
    
    return Command(
        goto="select_next_agent",  # Go to agent selector
        update={
            "messages": state["messages"] + [response],
            "available_findings": new_findings,
            "test_results": new_test_results,
        }
    )


def _format_available_findings(findings: Dict[str, Any]) -> str:
    """Format available clinical findings."""
    if not findings:
        return "No additional information disclosed yet."
    
    formatted = []
    for category, details in findings.items():
        formatted.append(f"- {category}: {details}")
    
    return "\n".join(formatted)


def _format_completed_tests(test_results: Dict[str, Any]) -> str:
    """Format completed test results."""
    completed = []
    for test, details in test_results.items():
        if details.get("results") and details["results"] != "pending":
            completed.append(f"- {test}: {details['results']}")
    
    return "\n".join(completed) if completed else "No tests completed yet."