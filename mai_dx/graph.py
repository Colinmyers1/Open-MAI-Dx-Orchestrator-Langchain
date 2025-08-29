"""
Graph construction for MAI Diagnostic Orchestrator using LangGraph.

This module builds the diagnostic workflow as a StateGraph, defining
the nodes, edges, and control flow for the multi-agent diagnostic process.
"""

from typing import Any, Dict, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic  
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from .state import DiagnosticState, AgentRole
from .agents import (
    hypothesis_agent, consensus_agent, gatekeeper_agent,
    test_chooser_agent, challenger_agent, stewardship_agent,
    checklist_agent, judge_agent
)
from .tools import (
    update_differential_diagnosis, make_consensus_decision,
    execute_diagnostic_test, request_information,
    DIAGNOSTIC_TOOLS
)


def create_model(model_name: str) -> BaseChatModel:
    """
    Create a language model instance based on the model name.
    
    Args:
        model_name: Name of the model to create
        
    Returns:
        Configured language model instance
    """
    model_name_lower = model_name.lower()
    
    if "gpt" in model_name_lower or "openai" in model_name_lower:
        return ChatOpenAI(model=model_name, temperature=0)
    elif "claude" in model_name_lower or "anthropic" in model_name_lower:
        return ChatAnthropic(model=model_name, temperature=0)
    elif "gemini" in model_name_lower or "google" in model_name_lower:
        # Handle different Gemini model name formats
        if model_name.startswith("gemini/"):
            clean_model = model_name.replace("gemini/", "")
        else:
            clean_model = model_name
        return ChatGoogleGenerativeAI(model=clean_model, temperature=0)
    else:
        # Default to GPT-4o-mini
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def select_next_agent(state: DiagnosticState) -> Dict[str, Any]:
    """
    Select the next specialist agent to consult based on current state.
    
    Args:
        state: Current diagnostic state
        
    Returns:
        State update with selected agent information
    """
    iteration = state.get("iteration", 0)
    
    # Round-robin selection with some intelligence
    agents = ["hypothesis", "test_chooser", "challenger", "stewardship", "checklist"]
    
    # Always start with hypothesis
    if iteration == 0 or not state.get("differential_diagnosis"):
        selected_agent = "hypothesis"
    else:
        # If we have a differential, get other perspectives
        agent_index = iteration % len(agents)
        
        # Skip hypothesis if we already have a recent differential
        if agents[agent_index] == "hypothesis" and state.get("hypothesis_analysis"):
            agent_index = (agent_index + 1) % len(agents)
        
        selected_agent = agents[agent_index]
    
    return {"selected_agent": selected_agent}


def route_consensus_decision(state: DiagnosticState) -> str:
    """
    Route based on the consensus decision.
    
    Args:
        state: Current diagnostic state
        
    Returns:
        Next node to visit based on consensus decision
    """
    current_action = state.get("current_action")
    
    if not current_action:
        return "gatekeeper"  # Default to getting more information
    
    # Handle both DiagnosticAction objects and dictionaries (due to serialization)
    if isinstance(current_action, dict):
        action_type = current_action.get("action_type")
    else:
        action_type = current_action.action_type
    
    if action_type == "ask":
        return "gatekeeper"
    elif action_type == "test":
        return "execute_test"
    elif action_type == "diagnose":
        return "finalize_diagnosis"
    else:
        return "gatekeeper"  # Default


def route_selected_agent(state: DiagnosticState) -> str:
    """
    Route to the selected agent.
    
    Args:
        state: Current diagnostic state
        
    Returns:
        Name of the selected agent
    """
    return state.get("selected_agent", "hypothesis")


def should_continue(state: DiagnosticState) -> str:
    """
    Determine if we should continue the diagnostic process or finalize.
    
    Args:
        state: Current diagnostic state
        
    Returns:
        Next step in the process
    """
    # Check if we're ready for diagnosis
    if state.get("ready_for_diagnosis"):
        return "finalize_diagnosis"
    
    # Check iteration limit
    if state.get("iteration", 0) >= state.get("max_iterations", 10):
        return "finalize_diagnosis"
    
    # Check budget (if enabled)
    if state.get("enable_budget_tracking"):
        remaining = state.get("current_budget", 0) - state.get("cumulative_cost", 0)
        if remaining <= 0:
            return "finalize_diagnosis"
    
    # Continue with next agent
    return "select_next_agent"


def execute_test_node(state: DiagnosticState) -> Command:
    """
    Execute a diagnostic test based on the current action.
    
    Args:
        state: Current diagnostic state
        
    Returns:
        Command to continue workflow
    """
    current_action = state.get("current_action")
    
    # Handle both DiagnosticAction objects and dictionaries (due to serialization)
    if isinstance(current_action, dict):
        action_type = current_action.get("action_type")
        test_name = current_action.get("content")
    else:
        action_type = current_action.action_type if current_action else None
        test_name = current_action.content if current_action else None
    
    if not current_action or action_type != "test":
        return Command(
            goto="consensus",
            update={"messages": state["messages"] + [SystemMessage(content="No test specified")]}
        )
    test_cost = 150  # Default cost, would be looked up from cost database
    
    # Check budget
    if state.get("enable_budget_tracking"):
        new_cost = state.get("cumulative_cost", 0) + test_cost
        if new_cost > state.get("current_budget", 0):
            return Command(
                goto="consensus",
                update={
                    "messages": state["messages"] + [SystemMessage(
                        content=f"Budget exceeded: Cannot order {test_name}"
                    )]
                }
            )
    
    # Execute test (simplified)
    test_result = f"Test {test_name} completed with findings"
    new_test_results = state.get("test_results", {}).copy()
    new_test_results[test_name] = {"cost": test_cost, "results": test_result}
    
    return Command(
        goto="gatekeeper",
        update={
            "messages": state["messages"] + [SystemMessage(
                content=f"Executed test: {test_name}"
            )],
            "test_results": new_test_results,
            "cumulative_cost": state.get("cumulative_cost", 0) + test_cost,
        }
    )


def finalize_diagnosis_node(state: DiagnosticState) -> Command:
    """
    Finalize the diagnosis and prepare for evaluation.
    
    Args:
        state: Current diagnostic state
        
    Returns:
        Command to proceed to evaluation
    """
    # Get final diagnosis from current action or top differential
    current_action = state.get("current_action")
    
    # Handle both DiagnosticAction objects and dictionaries (due to serialization)
    if isinstance(current_action, dict):
        action_type = current_action.get("action_type")
        content = current_action.get("content")
    else:
        action_type = current_action.action_type if current_action else None
        content = current_action.content if current_action else None
    
    if current_action and action_type == "diagnose":
        final_diagnosis = content
    elif state.get("differential_diagnosis"):
        # Use top diagnosis from differential
        top_diagnosis = max(state["differential_diagnosis"], key=lambda x: x.probability)
        final_diagnosis = top_diagnosis.diagnosis
    else:
        final_diagnosis = "Unable to reach definitive diagnosis"
    
    return Command(
        goto="judge",
        update={
            "messages": state["messages"] + [SystemMessage(
                content=f"Final diagnosis: {final_diagnosis}"
            )],
            "final_diagnosis": final_diagnosis,
        }
    )


def increment_iteration(state: DiagnosticState) -> Dict[str, Any]:
    """
    Increment the iteration counter and update state.
    
    Args:
        state: Current diagnostic state
        
    Returns:
        State update with incremented iteration
    """
    return {
        "iteration": state.get("iteration", 0) + 1
    }


def build_diagnostic_graph(model_name: str = "gpt-4o-mini") -> StateGraph:
    """
    Build the complete diagnostic workflow graph.
    
    Args:
        model_name: Name of the language model to use
        
    Returns:
        Compiled diagnostic StateGraph
    """
    # Create the language model
    model = create_model(model_name)
    
    # Create the graph
    graph = StateGraph(DiagnosticState)
    
    # Add agent nodes (with model binding)
    def make_agent_node(agent_func):
        def node(state: DiagnosticState) -> Command:
            return agent_func(state, model)
        return node
    
    graph.add_node("hypothesis", make_agent_node(hypothesis_agent))
    graph.add_node("test_chooser", make_agent_node(test_chooser_agent))
    graph.add_node("challenger", make_agent_node(challenger_agent))
    graph.add_node("stewardship", make_agent_node(stewardship_agent))
    graph.add_node("checklist", make_agent_node(checklist_agent))
    graph.add_node("consensus", make_agent_node(consensus_agent))
    graph.add_node("gatekeeper", make_agent_node(gatekeeper_agent))
    graph.add_node("judge", make_agent_node(judge_agent))
    
    # Add utility nodes
    graph.add_node("select_next_agent", select_next_agent)
    graph.add_node("execute_test", execute_test_node)
    graph.add_node("finalize_diagnosis", finalize_diagnosis_node)
    graph.add_node("increment_iteration", increment_iteration)
    
    # Add tool node for handling tool calls
    tool_node = ToolNode(DIAGNOSTIC_TOOLS)
    graph.add_node("tools", tool_node)
    
    # Define the workflow edges
    
    # Start with agent selection
    graph.add_edge(START, "select_next_agent")
    
    # Agent selection routes to specific agents
    graph.add_conditional_edges(
        "select_next_agent",
        route_selected_agent,
        {
            "hypothesis": "hypothesis",
            "test_chooser": "test_chooser", 
            "challenger": "challenger",
            "stewardship": "stewardship",
            "checklist": "checklist",
        }
    )
    
    # All specialist agents go to consensus
    for agent in ["hypothesis", "test_chooser", "challenger", "stewardship", "checklist"]:
        graph.add_edge(agent, "consensus")
    
    # Consensus routes based on decision
    graph.add_conditional_edges(
        "consensus",
        route_consensus_decision,
        {
            "gatekeeper": "gatekeeper",
            "execute_test": "execute_test", 
            "finalize_diagnosis": "finalize_diagnosis",
            END: END,
        }
    )
    
    # Gatekeeper goes to iteration increment
    graph.add_edge("gatekeeper", "increment_iteration")
    
    # Test execution goes to gatekeeper for results
    graph.add_edge("execute_test", "gatekeeper")
    
    # Iteration increment checks if we should continue
    graph.add_conditional_edges(
        "increment_iteration",
        should_continue,
        {
            "select_next_agent": "select_next_agent",
            "finalize_diagnosis": "finalize_diagnosis",
            END: END,
        }
    )
    
    # Finalization goes to judge
    graph.add_edge("finalize_diagnosis", "judge")
    
    # Judge ends the process
    graph.add_edge("judge", END)
    
    return graph


def compile_diagnostic_graph(
    model_name: str = "gpt-4o-mini",
    checkpointer: bool = True
) -> StateGraph:
    """
    Build and compile the diagnostic workflow graph.
    
    Args:
        model_name: Name of the language model to use
        checkpointer: Whether to enable checkpointing for state persistence
        
    Returns:
        Compiled and ready-to-use diagnostic graph
    """
    graph = build_diagnostic_graph(model_name)
    
    # Add checkpointer if requested
    if checkpointer:
        memory = MemorySaver()
        return graph.compile(checkpointer=memory)
    else:
        return graph.compile()