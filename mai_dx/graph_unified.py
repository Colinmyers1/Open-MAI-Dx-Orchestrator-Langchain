"""
Simplified Graph for Paper-Aligned Sequential Diagnosis.

This module implements the Sequential Diagnosis methodology from the Microsoft Research paper
using a single Diagnostic Agent that internally coordinates a virtual panel of specialists
via LangGraph Swarm.
"""

import os
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
from loguru import logger

from .state import DiagnosticState
from .agents.diagnostic_agent import unified_diagnostic_agent
from .user_interaction import user_interaction_node
from .tools import (
    execute_diagnostic_test, request_information,
    DIAGNOSTIC_TOOLS, make_consensus_decision
)
from .gatekeeper import GatekeeperAgent
from .judge import JudgeAgent


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
        
        # Map newer models to stable versions for LangGraph compatibility
        model_mapping = {
            "gemini-2.5-pro": "gemini-1.5-pro",
            "gemini-2.5-flash": "gemini-1.5-flash", 
            "gemini-2.0-flash": "gemini-2.0-flash-001",
            "gemini-2.0-flash-001": "gemini-2.0-flash-001"
        }
        
        clean_model = model_mapping.get(clean_model, clean_model)
        
        return ChatGoogleGenerativeAI(
            model=clean_model,
            temperature=0,
            convert_system_message_to_human=True
        )
    else:
        # Default to Gemini 2.0 Flash (latest)
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-001",
            temperature=0,
            convert_system_message_to_human=True
        )


def route_action_type(state: DiagnosticState) -> Literal["ask", "test", "diagnose", "error"]:
    """
    Route based on the diagnostic agent's decision.
    
    Args:
        state: Current diagnostic state
        
    Returns:
        Next node to route to based on action type
    """
    current_action = state.get("current_action", {})
    action_type = current_action.get("action_type", "error")
    
    logger.info(f"🔀 Routing action: {action_type}")
    
    if action_type in ["ask", "test", "diagnose"]:
        return action_type
    else:
        logger.warning(f"⚠️ Unknown action type: {action_type}, routing to error")
        return "error"


def should_terminate(state: DiagnosticState) -> bool:
    """
    Determine if the diagnostic process should terminate.
    
    Args:
        state: Current diagnostic state
        
    Returns:
        True if process should terminate
    """
    # Check workflow completion flag first
    if state.get("workflow_complete", False):
        logger.info("✅ Workflow marked as complete")
        return True
    
    # Check if we have a diagnosis ready
    if state.get("ready_for_diagnosis", False):
        logger.info("🏁 Ready for diagnosis - terminating")
        return True
    
    # Safety check for maximum iterations - force diagnosis
    iteration = state.get("iteration", 0)
    if iteration >= 10:  # Reduced from 50
        logger.warning(f"⚠️ Maximum iterations ({iteration}) reached - forcing diagnosis")
        # Set final diagnosis if not already set
        if not state.get("final_diagnosis"):
            import time
            return True  # This will trigger termination and the main.py should handle diagnosis
        return True
    
    # Check stagnation
    stagnation = state.get("stagnation_counter", 0)
    if stagnation >= 5:  # Force termination after stagnation
        logger.warning(f"⚠️ Stagnation detected ({stagnation}), forcing termination")
        return True
    
    logger.info(f"🔄 Continuing (iteration {iteration}, stagnation {stagnation})")
    return False


def execute_test_node(state: DiagnosticState) -> Dict[str, Any]:
    """
    Execute a diagnostic test based on the current action.
    
    Args:
        state: Current diagnostic state
        
    Returns:
        State update with test results
    """
    current_action = state.get("current_action", {})
    test_content = current_action.get("content", "")
    
    logger.info(f"🧪 Executing test: {test_content}")
    
    # Extract test from XML tags
    import re
    test_matches = re.findall(r'<test>(.*?)</test>', test_content, re.DOTALL)
    tests_to_run = test_matches if test_matches else [test_content.strip()]
    
    test_results = []
    total_cost = 0
    
    for test_name in tests_to_run:
        if test_name.strip():
            # Mock test execution - in real implementation would integrate with gatekeeper
            result = {
                "test_name": test_name.strip(),
                "result": f"Normal findings for {test_name.strip()}",  # Mock result
                "cost": 100,  # Mock cost
                "timestamp": "current"
            }
            test_results.append(result)
            total_cost += result["cost"]
    
    # Update state - ensure existing_results is always a list
    existing_results = state.get("test_results", [])
    if isinstance(existing_results, dict):
        existing_results = []  # Reset to empty list if it's a dict
    existing_cost = state.get("cumulative_cost", 0)
    
    return {
        "test_results": existing_results + test_results,
        "cumulative_cost": existing_cost + total_cost,
        "last_action_type": "test"
    }


def ask_question_node(state: DiagnosticState) -> Dict[str, Any]:
    """
    Process a question request by interacting with the gatekeeper.
    
    Args:
        state: Current diagnostic state
        
    Returns:
        State update with new information
    """
    current_action = state.get("current_action", {})
    question_content = current_action.get("content", "")
    
    logger.info(f"❓ Processing question: {question_content}")
    
    # Extract questions from XML tags
    import re
    question_matches = re.findall(r'<question>(.*?)</question>', question_content, re.DOTALL)
    questions = question_matches if question_matches else [question_content.strip()]
    
    # For now, return a mock response
    # In full implementation, this would interact with the gatekeeper
    new_info = f"Additional clinical information obtained: {', '.join(questions)}"
    
    existing_info = state.get("additional_clinical_info", [])
    if not isinstance(existing_info, list):
        existing_info = []  # Ensure it's always a list
    
    return {
        "additional_clinical_info": existing_info + [new_info],
        "cumulative_cost": state.get("cumulative_cost", 0) + 50,  # Cost for information gathering
        "last_action_type": "ask"
    }


def diagnose_node(state: DiagnosticState) -> Dict[str, Any]:
    """
    Process a final diagnosis.
    
    Args:
        state: Current diagnostic state
        
    Returns:
        State update with final diagnosis
    """
    current_action = state.get("current_action", {})
    diagnosis_content = current_action.get("content", "")
    
    logger.info(f"🎯 Processing diagnosis: {diagnosis_content}")
    
    # Extract diagnosis from XML tags
    import re
    diagnosis_matches = re.findall(r'<diagnosis>(.*?)</diagnosis>', diagnosis_content, re.DOTALL)
    final_diagnosis = diagnosis_matches[0] if diagnosis_matches else diagnosis_content.strip()
    
    return {
        "final_diagnosis": final_diagnosis,
        "workflow_complete": True,
        "ready_for_diagnosis": True,
        "last_action_type": "diagnose"
    }


def increment_iteration(state: DiagnosticState) -> Dict[str, Any]:
    """
    Increment the iteration counter and update workflow state.
    
    Args:
        state: Current diagnostic state
        
    Returns:
        State update with incremented iteration
    """
    current_iteration = state.get("iteration", 0)
    new_iteration = current_iteration + 1
    
    logger.info(f"🔢 Incrementing iteration: {current_iteration} -> {new_iteration}")
    
    return {"iteration": new_iteration}


def error_handler_node(state: DiagnosticState) -> Dict[str, Any]:
    """
    Handle errors in the diagnostic process.
    
    Args:
        state: Current diagnostic state
        
    Returns:
        State update with error handling
    """
    logger.error("❌ Error in diagnostic process, forcing completion")
    
    return {
        "final_diagnosis": "Diagnostic process encountered an error. Clinical assessment needed.",
        "workflow_complete": True,
        "ready_for_diagnosis": True
    }


def build_unified_diagnostic_graph(model_name: str = "gemini-2.0-flash-001") -> StateGraph:
    """
    Build the unified diagnostic workflow graph following the Sequential Diagnosis paper.
    
    This implementation uses a single Diagnostic Agent that internally coordinates
    a virtual panel of specialists using LangGraph Swarm, matching the paper's
    architecture where one agent makes structured decisions per turn.
    
    Args:
        model_name: Name of the language model to use
        
    Returns:
        Compiled diagnostic StateGraph
    """
    logger.info(f"🏗️ Building unified diagnostic graph with model: {model_name}")
    
    # Create the language model
    model = create_model(model_name)
    
    # Create the graph
    graph = StateGraph(DiagnosticState)
    
    # Add the unified diagnostic agent node
    def make_diagnostic_agent_node(state: DiagnosticState) -> Dict[str, Any]:
        return unified_diagnostic_agent(state, model)
    
    graph.add_node("diagnostic_agent", make_diagnostic_agent_node)
    
    # Add action execution nodes
    graph.add_node("ask", ask_question_node)
    graph.add_node("test", execute_test_node)
    graph.add_node("diagnose", diagnose_node)
    graph.add_node("increment_iteration", increment_iteration)
    graph.add_node("error", error_handler_node)
    
    # Add tool node for consensus decisions
    tool_node = ToolNode([make_consensus_decision])
    graph.add_node("tools", tool_node)
    
    # Define the workflow edges
    
    # Start with diagnostic agent
    graph.add_edge(START, "diagnostic_agent")
    
    # Diagnostic agent can call tools or route to actions
    graph.add_conditional_edges(
        "diagnostic_agent",
        lambda state: "tools" if state.get("messages", []) and 
                      hasattr(state["messages"][-1], "tool_calls") and 
                      state["messages"][-1].tool_calls else route_action_type(state),
        {
            "tools": "tools", 
            "ask": "ask",
            "test": "test", 
            "diagnose": "diagnose",
            "error": "error"
        }
    )
    
    # Tools route back to diagnostic agent  
    graph.add_edge("tools", "diagnostic_agent")
    
    # Action nodes increment iteration and route back
    graph.add_edge("ask", "increment_iteration")
    graph.add_edge("test", "increment_iteration") 
    
    # Iteration increment routes to termination check or back to diagnostic agent
    graph.add_conditional_edges(
        "increment_iteration", 
        should_terminate,
        {
            True: END,
            False: "diagnostic_agent"
        }
    )
    
    # Diagnosis and errors terminate
    graph.add_edge("diagnose", END)
    graph.add_edge("error", END)
    
    logger.info("✅ Unified diagnostic graph construction complete")
    
    return graph


def compile_unified_diagnostic_graph(model_name: str = "gemini-2.0-flash-001") -> Any:
    """
    Compile the unified diagnostic graph with checkpointing.
    
    Args:
        model_name: Name of the language model to use
        
    Returns:
        Compiled graph ready for execution
    """
    graph = build_unified_diagnostic_graph(model_name)
    
    # Add checkpointer for state persistence
    checkpointer = MemorySaver()
    
    # Compile with reduced recursion limit
    compiled_graph = graph.compile(
        checkpointer=checkpointer,
        debug=bool(os.getenv("MAIDX_DEBUG", False))
    )
    
    logger.info("📦 Unified diagnostic graph compiled successfully")
    
    return compiled_graph