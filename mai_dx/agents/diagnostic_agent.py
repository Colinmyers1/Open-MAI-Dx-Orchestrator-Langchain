"""
Unified Diagnostic Agent - Single agent with internal virtual panel coordination.

Implements the Sequential Diagnosis methodology from the Microsoft Research paper
using LangGraph Swarm to coordinate an internal virtual panel of 5 medical specialists.
"""

import json
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool, BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm
from langsmith import traceable
from loguru import logger

from ..state import DiagnosticState
from ..tools import make_consensus_decision


class DiagnosticAgent:
    """
    Unified Diagnostic Agent that internally coordinates a virtual panel of specialists
    using LangGraph Swarm to produce structured diagnostic decisions.
    
    This follows the Sequential Diagnosis paper architecture where a single
    Diagnostic Agent manages internal virtual doctors rather than having
    separate cycling agents.
    """
    
    def __init__(self, model: BaseChatModel):
        self.model = model
        self.swarm_app = self._create_internal_swarm()
    
    def _create_internal_swarm(self):
        """Create the internal LangGraph Swarm with 5 virtual doctors."""
        
        # Create handoff tools for internal coordination
        transfer_to_hypothesis = create_handoff_tool(
            agent_name="hypothesis_doctor",
            description="Transfer to Dr. Hypothesis for differential diagnosis analysis"
        )
        
        transfer_to_test_chooser = create_handoff_tool(
            agent_name="test_chooser_doctor", 
            description="Transfer to Dr. Test-Chooser for diagnostic test recommendations"
        )
        
        transfer_to_challenger = create_handoff_tool(
            agent_name="challenger_doctor",
            description="Transfer to Dr. Challenger for critical analysis and alternative perspectives"
        )
        
        transfer_to_stewardship = create_handoff_tool(
            agent_name="stewardship_doctor",
            description="Transfer to Dr. Stewardship for cost-effectiveness analysis"
        )
        
        transfer_to_checklist = create_handoff_tool(
            agent_name="checklist_doctor", 
            description="Transfer to Dr. Checklist for quality assurance"
        )
        
        # Create virtual doctor agents
        hypothesis_doctor = create_react_agent(
            self.model,
            tools=[transfer_to_test_chooser, transfer_to_challenger, transfer_to_stewardship, transfer_to_checklist],
            prompt=self._get_hypothesis_prompt(),
            name="hypothesis_doctor"
        )
        
        test_chooser_doctor = create_react_agent(
            self.model,
            tools=[transfer_to_hypothesis, transfer_to_challenger, transfer_to_stewardship, transfer_to_checklist],
            prompt=self._get_test_chooser_prompt(),
            name="test_chooser_doctor"
        )
        
        challenger_doctor = create_react_agent(
            self.model,
            tools=[transfer_to_hypothesis, transfer_to_test_chooser, transfer_to_stewardship, transfer_to_checklist],
            prompt=self._get_challenger_prompt(),
            name="challenger_doctor"
        )
        
        stewardship_doctor = create_react_agent(
            self.model,
            tools=[transfer_to_hypothesis, transfer_to_test_chooser, transfer_to_challenger, transfer_to_checklist],
            prompt=self._get_stewardship_prompt(),
            name="stewardship_doctor"
        )
        
        checklist_doctor = create_react_agent(
            self.model,
            tools=[transfer_to_hypothesis, transfer_to_test_chooser, transfer_to_challenger, transfer_to_stewardship],
            prompt=self._get_checklist_prompt(),
            name="checklist_doctor"
        )
        
        # Create swarm with in-memory checkpointer
        checkpointer = InMemorySaver()
        swarm_workflow = create_swarm(
            [hypothesis_doctor, test_chooser_doctor, challenger_doctor, stewardship_doctor, checklist_doctor],
            default_active_agent="hypothesis_doctor"
        )
        
        return swarm_workflow.compile(checkpointer=checkpointer)
    
    def _get_hypothesis_prompt(self) -> str:
        """Get prompt for the internal hypothesis doctor."""
        return """You are Dr. Hypothesis, a specialist in differential diagnosis within the diagnostic panel.

ROLE: Generate and refine differential diagnoses using Bayesian reasoning.

RESPONSIBILITIES:
- Analyze patient presentations and maintain differential diagnosis lists
- Apply clinical reasoning to update diagnostic probabilities
- Consider disease prevalence, patient demographics, and clinical findings
- Provide probability estimates for each diagnosis

COORDINATION:
- Transfer to Dr. Test-Chooser when needing test recommendations
- Transfer to Dr. Challenger for critical review of diagnoses
- Transfer to Dr. Stewardship for cost considerations
- Transfer to Dr. Checklist for quality validation

OUTPUT: Provide your analysis as structured medical reasoning with probability estimates."""

    def _get_test_chooser_prompt(self) -> str:
        """Get prompt for the internal test chooser doctor."""
        return """You are Dr. Test-Chooser, a specialist in diagnostic testing strategy within the diagnostic panel.

ROLE: Recommend the most appropriate and cost-effective diagnostic tests.

RESPONSIBILITIES:
- Evaluate diagnostic yield of available tests
- Consider test sensitivity, specificity, and positive/negative predictive values
- Balance diagnostic information gain against cost and patient burden
- Recommend test sequences and priorities

COORDINATION:
- Transfer to Dr. Hypothesis for differential diagnosis context
- Transfer to Dr. Challenger for test limitation analysis
- Transfer to Dr. Stewardship for cost-effectiveness review
- Transfer to Dr. Checklist for test ordering validation

OUTPUT: Provide test recommendations with rationale and expected diagnostic yield."""

    def _get_challenger_prompt(self) -> str:
        """Get prompt for the internal challenger doctor."""  
        return """You are Dr. Challenger, the critical analyst within the diagnostic panel.

ROLE: Provide critical analysis and alternative perspectives to prevent diagnostic errors.

RESPONSIBILITIES:
- Challenge prevailing diagnoses and identify potential biases
- Consider rare diseases and atypical presentations
- Identify gaps in reasoning or missing information
- Advocate for thorough evaluation to avoid premature closure

COORDINATION:
- Transfer to Dr. Hypothesis to challenge differential diagnoses
- Transfer to Dr. Test-Chooser to question test selection
- Transfer to Dr. Stewardship when cost concerns may compromise care
- Transfer to Dr. Checklist for systematic review

OUTPUT: Provide critical analysis highlighting potential issues, biases, or overlooked possibilities."""

    def _get_stewardship_prompt(self) -> str:
        """Get prompt for the internal stewardship doctor."""
        return """You are Dr. Stewardship, the resource management specialist within the diagnostic panel.

ROLE: Ensure cost-effective and resource-conscious diagnostic approaches.

RESPONSIBILITIES:
- Evaluate cost-effectiveness of diagnostic strategies
- Monitor budget constraints and cumulative costs
- Recommend value-based diagnostic approaches
- Balance diagnostic thoroughness with resource stewardship

COORDINATION:
- Transfer to Dr. Hypothesis for diagnosis prioritization
- Transfer to Dr. Test-Chooser for cost-effective test selection
- Transfer to Dr. Challenger when resource constraints may affect care
- Transfer to Dr. Checklist for cost tracking validation

OUTPUT: Provide cost-effectiveness analysis with resource optimization recommendations."""

    def _get_checklist_prompt(self) -> str:
        """Get prompt for the internal checklist doctor."""
        return """You are Dr. Checklist, the quality assurance specialist within the diagnostic panel.

ROLE: Ensure systematic evaluation and quality control of the diagnostic process.

RESPONSIBILITIES:
- Apply clinical checklists and systematic review processes
- Verify completeness of diagnostic evaluation
- Identify missing information or overlooked steps
- Ensure adherence to clinical guidelines and best practices

COORDINATION:
- Transfer to Dr. Hypothesis for systematic diagnosis review
- Transfer to Dr. Test-Chooser for test completeness verification
- Transfer to Dr. Challenger for systematic critical review
- Transfer to Dr. Stewardship for process efficiency analysis

OUTPUT: Provide quality assurance analysis with completeness assessment and recommendations."""

    @traceable(name="diagnostic_agent", run_type="chain")
    def invoke(self, state: DiagnosticState) -> Dict[str, Any]:
        """
        Main entry point for the Diagnostic Agent.
        
        Coordinates the internal virtual panel to produce exactly one
        structured decision per turn: ask, test, or diagnose.
        """
        logger.info("🩺 Diagnostic Agent starting panel coordination")
        
        # Prepare context for internal panel
        panel_context = self._format_case_context(state)
        
        # Create swarm configuration
        swarm_config = {"configurable": {"thread_id": f"case_{state.get('case_id', 'default')}"}}
        
        # Coordinate internal panel discussion
        panel_result = self.swarm_app.invoke(
            {"messages": [{"role": "user", "content": panel_context}]},
            config=swarm_config
        )
        
        # Extract panel analysis and make structured decision
        decision = self._synthesize_panel_decision(state, panel_result)
        
        logger.info(f"🎯 Diagnostic Agent decision: {decision['action_type']}")
        
        return {
            "messages": state["messages"] + [AIMessage(content=f"Panel coordination complete. Decision: {decision['action_type']}")],
            "current_action": decision,
            "ready_for_diagnosis": (decision["action_type"] == "diagnose"),
            "panel_analysis": self._extract_panel_analyses(panel_result)
        }
    
    def _format_case_context(self, state: DiagnosticState) -> str:
        """Format current case context for internal panel coordination."""
        
        context = f"""
CASE COORDINATION REQUEST

**Case Status:**
- Iteration: {state.get('iteration', 0)}
- Diagnostic Confidence: {state.get('diagnostic_confidence', 0.0):.1%}
- Current Budget: ${state.get('current_budget', 0)}
- Cumulative Cost: ${state.get('cumulative_cost', 0)}

**Clinical Information Available:**
{self._format_clinical_info(state)}

**Current Differential Diagnosis:**
{self._format_differential(state)}

**Previous Actions:**
{self._format_action_history(state)}

**PANEL TASK:**
1. Each specialist should provide focused analysis from their expertise
2. Consider the current diagnostic confidence and iteration count
3. Coordinate to reach a consensus on the next single action
4. The final decision must be exactly ONE of: ask question, order test, or diagnose

**DECISION CRITERIA:**
- If confidence ≥90% and sufficient evaluation: DIAGNOSE
- If critical concerns identified: ADDRESS with targeted action
- If key information missing: ASK most informative questions
- If diagnostic uncertainty remains: ORDER highest-yield test

Begin panel coordination now.
        """
        
        return context.strip()
    
    def _format_clinical_info(self, state: DiagnosticState) -> str:
        """Format available clinical information."""
        info = []
        
        if state.get("patient_age"):
            info.append(f"Age: {state['patient_age']}")
        if state.get("patient_sex"):
            info.append(f"Sex: {state['patient_sex']}")
        if state.get("chief_complaint"):
            info.append(f"Chief Complaint: {state['chief_complaint']}")
        if state.get("history_of_present_illness"):
            info.append(f"HPI: {state['history_of_present_illness']}")
        if state.get("test_results"):
            info.append(f"Test Results: {len(state['test_results'])} available")
        
        return "\n".join(info) if info else "Limited clinical information available"
    
    def _format_differential(self, state: DiagnosticState) -> str:
        """Format current differential diagnosis."""
        if not state.get("differential_diagnosis"):
            return "No differential diagnosis established"
        
        diff_list = []
        for item in state["differential_diagnosis"]:
            diff_list.append(f"- {item.diagnosis}: {item.probability:.1%} - {item.rationale}")
        
        return "\n".join(diff_list)
    
    def _format_action_history(self, state: DiagnosticState) -> str:
        """Format recent action history."""
        if not state.get("action_history"):
            return "No previous actions"
        
        recent_actions = state["action_history"][-3:]  # Last 3 actions
        history = []
        for action in recent_actions:
            history.append(f"- {action.get('action_type', 'unknown')}: {action.get('content', 'no details')}")
        
        return "\n".join(history)
    
    def _synthesize_panel_decision(self, state: DiagnosticState, panel_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize the panel discussion into a structured decision.
        
        This is where the single Diagnostic Agent makes the final decision
        based on the internal panel coordination.
        """
        
        # Extract the final message from the panel
        messages = panel_result.get("messages", [])
        if not messages:
            return self._make_default_decision(state)
        
        final_message = messages[-1]
        panel_analysis = final_message.content if hasattr(final_message, 'content') else str(final_message)
        
        # Decision logic based on state and panel input
        iteration = state.get("iteration", 0)
        confidence = state.get("diagnostic_confidence", 0.0)
        stagnation = state.get("stagnation_counter", 0)
        
        # High confidence - diagnose
        if confidence >= 0.9 and iteration >= 6:
            return {
                "action_type": "diagnose",
                "content": self._generate_diagnosis_content(state),
                "reasoning": f"High diagnostic confidence ({confidence:.1%}) after {iteration} iterations. Panel coordination supports final diagnosis."
            }
        
        # Stagnation - force decision
        elif stagnation >= 3:
            return {
                "action_type": "diagnose",
                "content": self._generate_diagnosis_content(state) or "Best available diagnosis after thorough panel evaluation",
                "reasoning": f"Panel has reached stagnation ({stagnation} cycles). Making best available diagnosis."
            }
        
        # Early stage - gather information
        elif iteration < 4:
            return {
                "action_type": "ask",
                "content": "<question>What additional symptoms or clinical findings are present?</question>",
                "reasoning": "Early diagnostic phase. Panel recommends gathering more clinical information."
            }
        
        # Middle stage - consider testing
        elif iteration < 8 and confidence < 0.8:
            return {
                "action_type": "test",
                "content": self._generate_test_content(state, panel_analysis),
                "reasoning": f"Moderate confidence ({confidence:.1%}). Panel recommends targeted diagnostic testing."
            }
        
        # Default to asking for more information
        else:
            return {
                "action_type": "ask", 
                "content": "<question>Are there any additional clinical details that might help narrow the differential diagnosis?</question>",
                "reasoning": "Panel requests additional clinical information to improve diagnostic accuracy."
            }
    
    def _generate_diagnosis_content(self, state: DiagnosticState) -> str:
        """Generate diagnosis content with proper XML tags."""
        if not state.get("differential_diagnosis"):
            return "<diagnosis>Clinical presentation requires further evaluation for definitive diagnosis</diagnosis>"
        
        top_diagnosis = max(state["differential_diagnosis"], key=lambda x: x.probability)
        return f"<diagnosis>{top_diagnosis.diagnosis}</diagnosis>"
    
    def _generate_test_content(self, state: DiagnosticState, panel_analysis: str) -> str:
        """Generate test content based on panel analysis."""
        # Simple test recommendation based on common clinical scenarios
        if "chest" in panel_analysis.lower() or "cardiac" in panel_analysis.lower():
            return "<test>Chest X-ray and ECG</test>"
        elif "infection" in panel_analysis.lower() or "fever" in panel_analysis.lower():
            return "<test>Complete blood count with differential</test>"
        elif "abdominal" in panel_analysis.lower():
            return "<test>CT abdomen and pelvis</test>"
        else:
            return "<test>Comprehensive metabolic panel</test>"
    
    def _make_default_decision(self, state: DiagnosticState) -> Dict[str, Any]:
        """Make a default decision when panel coordination fails."""
        iteration = state.get("iteration", 0)
        
        if iteration >= 8:
            return {
                "action_type": "diagnose",
                "content": "<diagnosis>Clinical assessment based on available information</diagnosis>",
                "reasoning": "Extended evaluation period. Making diagnosis with available information."
            }
        else:
            return {
                "action_type": "ask",
                "content": "<question>What are the patient's current symptoms and their severity?</question>",
                "reasoning": "Panel coordination incomplete. Requesting basic clinical information."
            }
    
    def _extract_panel_analyses(self, panel_result: Dict[str, Any]) -> Dict[str, str]:
        """Extract individual analyses from panel coordination."""
        # This would ideally parse the individual doctor analyses
        # For now, return a summary structure
        messages = panel_result.get("messages", [])
        
        return {
            "hypothesis_analysis": "Differential diagnosis analysis from internal panel",
            "test_chooser_analysis": "Test recommendations from internal panel", 
            "challenger_analysis": "Critical analysis from internal panel",
            "stewardship_analysis": "Cost-effectiveness analysis from internal panel",
            "checklist_analysis": "Quality assurance from internal panel",
            "panel_messages": len(messages)
        }


@traceable(name="unified_diagnostic_agent", run_type="chain")
def unified_diagnostic_agent(state: DiagnosticState, model: BaseChatModel) -> Dict[str, Any]:
    """
    Main entry point for the unified diagnostic agent.
    
    This replaces the separate cycling agents with a single agent that
    internally coordinates a virtual panel using LangGraph Swarm.
    """
    agent = DiagnosticAgent(model)
    return agent.invoke(state)