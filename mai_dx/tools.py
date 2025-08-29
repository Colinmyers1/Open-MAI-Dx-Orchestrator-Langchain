"""
Tools for MAI Diagnostic Orchestrator using LangGraph.

This module defines the tools that agents can use during the diagnostic process,
including structured output tools, diagnostic test execution, and information gathering.
"""

from typing import Annotated, List, Dict, Any
from langchain_core.tools import tool, InjectedToolCallId, StructuredTool
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from .state import DiagnosticState, DifferentialDiagnosisItem, DiagnosticAction, get_test_cost


@tool
def update_differential_diagnosis(
    summary: str,
    differential_diagnoses: List[Dict[str, Any]],
    key_evidence: str,
    contradictory_evidence: str = "",
    state: Annotated[DiagnosticState, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """
    Update the differential diagnosis with structured probabilities and reasoning.
    
    Args:
        summary: One-sentence summary of primary diagnostic conclusion
        differential_diagnoses: List of diagnosis objects with diagnosis, probability, rationale
        key_evidence: Key supporting evidence for leading hypotheses
        contradictory_evidence: Critical contradictory evidence that must be addressed
        state: Current diagnostic state (injected)
        tool_call_id: Tool call ID (injected)
        
    Returns:
        Command to update state and continue workflow
    """
    # Convert to DifferentialDiagnosisItem objects
    diagnosis_items = []
    for item in differential_diagnoses:
        diagnosis_items.append(DifferentialDiagnosisItem(
            diagnosis=item["diagnosis"],
            probability=item["probability"], 
            rationale=item["rationale"]
        ))
    
    # Create tool response message
    tool_message = ToolMessage(
        content=f"Updated differential diagnosis: {summary}",
        name="update_differential_diagnosis",
        tool_call_id=tool_call_id,
    )
    
    # Update state with new differential
    state_update = {
        "messages": state["messages"] + [tool_message],
        "differential_diagnosis": diagnosis_items,
        "current_hypothesis": summary,
        "hypothesis_analysis": f"Summary: {summary}\nKey Evidence: {key_evidence}\nContradictory Evidence: {contradictory_evidence}",
    }
    
    return Command(
        goto="consensus",
        update=state_update
    )


@tool  
def make_consensus_decision(
    action_type: str,
    content: str, 
    reasoning: str,
    state: Annotated[DiagnosticState, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """
    Make a structured consensus decision for the next diagnostic action.
    
    Args:
        action_type: The type of action ("ask", "test", "diagnose")
        content: The specific content of the action 
        reasoning: The detailed reasoning behind this decision
        state: Current diagnostic state (injected)
        tool_call_id: Tool call ID (injected)
        
    Returns:
        Command to execute the decided action
    """
    # Create the diagnostic action
    action = DiagnosticAction(
        action_type=action_type,
        content=content,
        reasoning=reasoning
    )
    
    # Create tool response message
    tool_message = ToolMessage(
        content=f"Consensus decision: {action_type} - {content}",
        name="make_consensus_decision",
        tool_call_id=tool_call_id,
    )
    
    # Determine next node based on action type
    if action_type == "ask":
        next_node = "gatekeeper"
    elif action_type == "test":
        next_node = "execute_test"
    elif action_type == "diagnose":
        next_node = "finalize_diagnosis"
    else:
        next_node = "consensus"  # Fallback
    
    # Update state with consensus decision
    state_update = {
        "messages": state["messages"] + [tool_message],
        "current_action": action,
        "ready_for_diagnosis": (action_type == "diagnose"),
    }
    
    return Command(
        goto=next_node,
        update=state_update
    )


@tool
def execute_diagnostic_test(
    test_name: str,
    state: Annotated[DiagnosticState, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """
    Execute a diagnostic test and update costs.
    
    Args:
        test_name: Name of the test to execute
        state: Current diagnostic state (injected)
        tool_call_id: Tool call ID (injected)
        
    Returns:
        Command to continue with gatekeeper for test results
    """
    # Calculate test cost
    test_cost = get_test_cost(test_name)
    new_cumulative_cost = state["cumulative_cost"] + test_cost
    
    # Check budget if enabled
    if state["enable_budget_tracking"] and new_cumulative_cost > state["current_budget"]:
        tool_message = ToolMessage(
            content=f"BUDGET EXCEEDED: Cannot order {test_name} (${test_cost}). Current cost: ${state['cumulative_cost']}, Budget: ${state['current_budget']}",
            name="execute_diagnostic_test",
            tool_call_id=tool_call_id,
        )
        
        state_update = {
            "messages": state["messages"] + [tool_message],
        }
        
        return Command(
            goto="consensus",  # Return to consensus to pick different action
            update=state_update
        )
    
    # Execute the test (add to test results)
    tool_message = ToolMessage(
        content=f"Ordered: {test_name} (Cost: ${test_cost})",
        name="execute_diagnostic_test",
        tool_call_id=tool_call_id,
    )
    
    # Update state with test execution
    new_test_results = state["test_results"].copy()
    new_test_results[test_name] = {"cost": test_cost, "ordered": True, "results": "pending"}
    
    state_update = {
        "messages": state["messages"] + [tool_message],
        "test_results": new_test_results,
        "cumulative_cost": new_cumulative_cost,
    }
    
    return Command(
        goto="gatekeeper",  # Get test results from gatekeeper
        update=state_update
    )


@tool
def request_information(
    query: str,
    state: Annotated[DiagnosticState, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """
    Request specific clinical information from the gatekeeper.
    
    Args:
        query: The specific information being requested
        state: Current diagnostic state (injected) 
        tool_call_id: Tool call ID (injected)
        
    Returns:
        Command to get information from gatekeeper
    """
    tool_message = ToolMessage(
        content=f"Information requested: {query}",
        name="request_information", 
        tool_call_id=tool_call_id,
    )
    
    state_update = {
        "messages": state["messages"] + [tool_message],
    }
    
    return Command(
        goto="gatekeeper",
        update=state_update
    )


@tool
def evaluate_diagnosis_accuracy(
    final_diagnosis: str,
    ground_truth: str,
    reasoning: str,
    state: Annotated[DiagnosticState, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Dict[str, Any]:
    """
    Evaluate the accuracy of the final diagnosis against ground truth.
    
    Args:
        final_diagnosis: The diagnosis rendered by the panel
        ground_truth: The correct diagnosis
        reasoning: Reasoning for the accuracy assessment
        state: Current diagnostic state (injected)
        tool_call_id: Tool call ID (injected)
        
    Returns:
        Dictionary with accuracy score and reasoning
    """
    # Simple accuracy scoring (can be enhanced with more sophisticated matching)
    final_lower = final_diagnosis.lower().strip()
    truth_lower = ground_truth.lower().strip()
    
    if final_lower == truth_lower:
        score = 5.0  # Exact match
    elif any(word in final_lower for word in truth_lower.split()) or any(word in truth_lower for word in final_lower.split()):
        score = 3.0  # Partial match
    else:
        score = 1.0  # No match
        
    return {
        "accuracy_score": score,
        "accuracy_reasoning": reasoning,
        "is_correct": score >= 4.0,
        "final_diagnosis": final_diagnosis,
        "ground_truth": ground_truth,
    }


# Agent handoff tools for multi-agent coordination
def create_agent_handoff_tool(agent_name: str, description: str = None):
    """
    Create a handoff tool for transferring control between agents.
    
    Args:
        agent_name: Name of the agent to transfer to
        description: Description of when to use this tool
        
    Returns:
        LangChain tool for agent handoff
    """
    tool_name = f"transfer_to_{agent_name}"
    tool_description = description or f"Transfer control to {agent_name}"
    
    def handoff_function(
        message: str,
        state: Annotated[DiagnosticState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Transfer control to another agent with a message."""
        
        tool_message = ToolMessage(
            content=f"Transferred to {agent_name}: {message}",
            name=tool_name,
            tool_call_id=tool_call_id,
        )
        
        state_update = {
            "messages": state["messages"] + [tool_message],
        }
        
        return Command(
            goto=agent_name,
            update=state_update
        )
    
    return StructuredTool.from_function(
        func=handoff_function,
        name=tool_name,
        description=tool_description,
    )


# Pre-defined handoff tools for the main agents
transfer_to_hypothesis = create_agent_handoff_tool("hypothesis", "Get differential diagnosis analysis")
transfer_to_test_chooser = create_agent_handoff_tool("test_chooser", "Get test selection recommendations") 
transfer_to_challenger = create_agent_handoff_tool("challenger", "Get critical analysis and alternative perspectives")
transfer_to_stewardship = create_agent_handoff_tool("stewardship", "Get cost-effectiveness analysis")
transfer_to_checklist = create_agent_handoff_tool("checklist", "Get quality control review")
transfer_to_gatekeeper = create_agent_handoff_tool("gatekeeper", "Get clinical information")
transfer_to_consensus = create_agent_handoff_tool("consensus", "Return to consensus coordination")


# List of all available tools for easy import
DIAGNOSTIC_TOOLS = [
    update_differential_diagnosis,
    make_consensus_decision, 
    execute_diagnostic_test,
    request_information,
    evaluate_diagnosis_accuracy,
]

HANDOFF_TOOLS = [
    transfer_to_hypothesis,
    transfer_to_test_chooser,
    transfer_to_challenger, 
    transfer_to_stewardship,
    transfer_to_checklist,
    transfer_to_gatekeeper,
    transfer_to_consensus,
]

ALL_TOOLS = DIAGNOSTIC_TOOLS + HANDOFF_TOOLS