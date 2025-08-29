"""
Consensus Coordinator - Decision Making Agent.

Synthesizes input from all specialist agents and makes structured decisions
about the next diagnostic action.
"""

from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage 
from langchain_core.language_models import BaseChatModel
from langgraph.types import Command

from ..state import DiagnosticState, CaseState
from ..tools import make_consensus_decision


def get_consensus_prompt(state: DiagnosticState) -> str:
    """Get the system prompt for the consensus coordinator with dynamic context."""
    
    dynamic_context = ""
    
    # Add situational context based on state
    iteration = state.get("iteration", 0)
    if iteration > 5:
        dynamic_context += f"""
**SITUATIONAL CONTEXT: EXTENDED CASE**
This case has gone through {iteration} iterations. Focus on decisive actions that will lead to a definitive diagnosis rather than additional exploratory steps.
"""
    
    # Budget considerations
    if state["enable_budget_tracking"]:
        remaining_budget = state["current_budget"] - state["cumulative_cost"]
        if remaining_budget < 500:
            dynamic_context += f"""
**SITUATIONAL CONTEXT: CRITICAL BUDGET**
Only ${remaining_budget} remaining. Any further actions must be essential for diagnosis.
"""
    
    # Stagnation detection
    if state.get("stagnation_detected"):
        dynamic_context += """
**SITUATIONAL CONTEXT: STAGNATION DETECTED**
The panel is repeating actions or analysis. You MUST make a decisive choice or provide final diagnosis.
"""
    
    base_prompt = f"""{dynamic_context}

MANDATE: Decide the next diagnostic action by synthesizing all panel input.

ROLE: You are the Consensus Coordinator, responsible for making structured decisions that advance the diagnostic process.

DECISION FRAMEWORK:
1. **If confidence >85% AND no major objections** → diagnose
2. **If Challenger raised critical concerns** → address top concern with targeted action
3. **If diagnostic uncertainty remains** → order highest information-gain test (prefer cheaper options)
4. **If missing key clinical information** → ask the most informative question

SYNTHESIS APPROACH:
- Review Dr. Hypothesis's differential diagnosis and confidence levels
- Consider Dr. Test-Chooser's recommendations for diagnostic yield
- Address Dr. Challenger's concerns about biases or alternative diagnoses  
- Incorporate Dr. Stewardship's cost-effectiveness analysis
- Ensure Dr. Checklist's quality concerns are addressed

OUTPUT REQUIREMENTS:
- You MUST call make_consensus_decision() with structured action
- Choose action_type: "ask", "test", or "diagnose"
- Provide specific content (question text, test name, or diagnosis)
- Give clear reasoning that references panel input

DECISION PRIORITIES:
1. Patient safety and diagnostic accuracy
2. Information value and diagnostic yield
3. Cost-effectiveness and resource stewardship
4. Quality assurance and completeness"""
    
    return base_prompt


def format_panel_context(state: DiagnosticState) -> str:
    """Format the current panel analyses into a structured context."""
    
    context = "**Current Diagnostic Panel Analysis:**\n\n"
    
    # Hypothesis analysis
    if state.get("hypothesis_analysis"):
        context += f"**Differential Diagnosis (Dr. Hypothesis):**\n{state['hypothesis_analysis']}\n\n"
    
    # Current differential with probabilities
    if state.get("differential_diagnosis"):
        context += "**Current Differential:**\n"
        for item in state["differential_diagnosis"]:
            context += f"- {item.diagnosis}: {item.probability:.1%} - {item.rationale}\n"
        context += "\n"
    
    # Test recommendations
    if state.get("test_chooser_analysis"):
        context += f"**Test Recommendations (Dr. Test-Chooser):**\n{state['test_chooser_analysis']}\n\n"
    
    # Critical analysis
    if state.get("challenger_analysis"):
        context += f"**Critical Analysis (Dr. Challenger):**\n{state['challenger_analysis']}\n\n"
    
    # Cost analysis
    if state.get("stewardship_analysis"):
        context += f"**Cost Analysis (Dr. Stewardship):**\n{state['stewardship_analysis']}\n\n"
    
    # Quality control
    if state.get("checklist_analysis"):
        context += f"**Quality Control (Dr. Checklist):**\n{state['checklist_analysis']}\n\n"
    
    # Current status
    context += f"""**Current Status:**
- Iteration: {state.get('iteration', 0)}/{state.get('max_iterations', 10)}
- Cumulative Cost: ${state.get('cumulative_cost', 0)}
- Budget Remaining: ${state.get('current_budget', 0) - state.get('cumulative_cost', 0)}
- Mode: {state.get('mode', 'standard')}
"""
    
    return context


def consensus_agent(state: DiagnosticState, model: BaseChatModel) -> Command:
    """
    Consensus Coordinator agent for making structured diagnostic decisions.
    
    Args:
        state: Current diagnostic state
        model: Language model to use
        
    Returns:
        Command to execute the decided action
    """
    # Get the system prompt
    system_prompt = get_consensus_prompt(state)
    
    # Format panel context
    panel_context = format_panel_context(state)
    
    # Add instruction for structured decision
    decision_prompt = """
Based on this comprehensive panel input, use the make_consensus_decision function to provide your structured action.

Consider:
- What is the highest-confidence diagnosis and its probability?
- Are there unresolved critical concerns from the Challenger?
- What would be the highest-yield next step (question or test)?
- Are we at a decision point for final diagnosis?

Make your decision now."""
    
    # Bind the make_consensus_decision tool
    model_with_tools = model.bind_tools([make_consensus_decision])
    
    # Create the input messages
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=panel_context),
        HumanMessage(content=decision_prompt)
    ]
    
    # Add recent conversation context for continuity (temporarily disabled for debugging)
    # recent_messages = state["messages"][-2:] if len(state["messages"]) > 2 else state["messages"]
    # messages.extend(recent_messages)
    
    # Invoke the model
    response = model_with_tools.invoke(messages)
    
    # The tool will handle the state update and routing
    # If no tool was called, make a default decision
    if not response.tool_calls:
        # Default action based on state
        if _should_diagnose(state):
            action_type = "diagnose"
            content = _get_top_diagnosis(state)
        elif state.get("iteration", 0) >= state.get("max_iterations", 10):
            action_type = "diagnose" 
            content = _get_top_diagnosis(state) or "Unable to reach definitive diagnosis"
        else:
            action_type = "ask"
            content = "Please provide additional clinical information to help narrow the differential diagnosis."
        
        return Command(
            goto="gatekeeper" if action_type == "ask" else "finalize_diagnosis",
            update={
                "messages": state["messages"] + [response],
                "current_action": {
                    "action_type": action_type,
                    "content": content,
                    "reasoning": "Default decision due to no structured output"
                },
                "ready_for_diagnosis": (action_type == "diagnose"),
            }
        )
    
    # Tool execution will be handled by the graph
    return Command(
        goto="gatekeeper",  # Default next step, tool will override
        update={"messages": state["messages"] + [response]}
    )


def _should_diagnose(state: DiagnosticState) -> bool:
    """Determine if we should make a diagnosis based on current state."""
    if not state.get("differential_diagnosis"):
        return False
    
    # Check if top diagnosis has high confidence
    max_prob = max(item.probability for item in state["differential_diagnosis"])
    return max_prob >= 0.85


def _get_top_diagnosis(state: DiagnosticState) -> str:
    """Get the highest probability diagnosis."""
    if not state.get("differential_diagnosis"):
        return "No differential diagnosis available"
    
    top_item = max(state["differential_diagnosis"], key=lambda x: x.probability)
    return top_item.diagnosis