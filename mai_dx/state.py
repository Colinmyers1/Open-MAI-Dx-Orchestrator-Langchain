"""
State management for MAI Diagnostic Orchestrator using LangGraph.

This module defines the state structure used throughout the diagnostic process,
including messages, diagnosis tracking, cost management, and case information.
"""

from typing import Annotated, Dict, List, Optional, Any, Union
from typing_extensions import TypedDict
from enum import Enum
from dataclasses import dataclass, field

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field


class AgentRole(Enum):
    """Enumeration of roles for the virtual physician panel."""
    
    HYPOTHESIS = "Dr. Hypothesis"
    TEST_CHOOSER = "Dr. Test-Chooser"
    CHALLENGER = "Dr. Challenger"
    STEWARDSHIP = "Dr. Stewardship"
    CHECKLIST = "Dr. Checklist"
    CONSENSUS = "Consensus Coordinator"
    GATEKEEPER = "Gatekeeper"
    JUDGE = "Judge"


class DifferentialDiagnosisItem(BaseModel):
    """Single differential diagnosis item."""
    
    diagnosis: str = Field(description="The diagnosis name")
    probability: float = Field(ge=0, le=1, description="Probability as decimal (0.0-1.0)")
    rationale: str = Field(description="Brief rationale for this diagnosis")


class DiagnosticAction(BaseModel):
    """Represents a diagnostic action decided by the consensus agent."""
    
    action_type: str = Field(description="The type of action (ask, test, diagnose)")
    content: Union[str, List[str]] = Field(description="The content of the action")
    reasoning: str = Field(description="The reasoning behind this decision")


@dataclass
class CaseState:
    """Structured state management for diagnostic process."""
    
    initial_vignette: str
    full_case_details: str
    ground_truth: str
    current_budget: int
    max_iterations: int = 10
    iteration: int = 0
    
    # Agent analysis tracking
    hypothesis_analysis: Optional[str] = None
    test_chooser_analysis: Optional[str] = None
    challenger_analysis: Optional[str] = None
    stewardship_analysis: Optional[str] = None
    checklist_analysis: Optional[str] = None
    
    # State flags
    stagnation_detected: bool = False
    situational_context: Optional[str] = None
    
    def to_prompt_context(self) -> str:
        """Generate a structured prompt context from current state."""
        prompt = f"""**Current Diagnostic Panel Analysis:**

**Hypothesis Generation (Dr. Hypothesis):**
{self.hypothesis_analysis or 'Not yet evaluated'}

**Test Selection (Dr. Test-Chooser):**
{self.test_chooser_analysis or 'No tests proposed'}

**Critical Analysis (Dr. Challenger):**
{self.challenger_analysis or 'No concerns raised'}

**Cost Analysis (Dr. Stewardship):**
{self.stewardship_analysis or 'Not evaluated'}

**Quality Control (Dr. Checklist):**
{self.checklist_analysis or 'No issues noted'}
"""
        
        if self.stagnation_detected:
            prompt += "\n**STAGNATION DETECTED** - The panel is repeating actions. You MUST make a decisive choice or provide final diagnosis."
        
        if self.situational_context:
            prompt += f"\n**Situational Context:** {self.situational_context}"
        
        return prompt


class DiagnosticState(TypedDict):
    """
    The main state structure for the diagnostic process.
    
    This extends the basic MessagesState with diagnostic-specific fields
    for tracking the diagnostic process, costs, and agent analyses.
    """
    
    # Core messaging (LangGraph standard)
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Case information
    initial_vignette: str
    full_case_details: str  
    ground_truth: str
    
    # Diagnostic tracking
    differential_diagnosis: Optional[List[DifferentialDiagnosisItem]]
    current_hypothesis: Optional[str]
    final_diagnosis: Optional[str]
    
    # Cost and iteration tracking
    cumulative_cost: int
    current_budget: int
    iteration: int
    max_iterations: int
    
    # Test results and findings
    test_results: Dict[str, Any]
    available_findings: Dict[str, Any]
    
    # Agent analyses (for panel coordination)
    hypothesis_analysis: Optional[str]
    test_chooser_analysis: Optional[str]
    challenger_analysis: Optional[str]
    stewardship_analysis: Optional[str] 
    checklist_analysis: Optional[str]
    
    # Control flags
    stagnation_detected: bool
    ready_for_diagnosis: bool
    current_action: Optional[DiagnosticAction]
    
    # Mode and configuration
    mode: str  # "instant", "question_only", "budgeted", "no_budget", "ensemble"
    model_name: str
    
    # Operational settings
    physician_visit_cost: int
    enable_budget_tracking: bool


class DiagnosisResult(BaseModel):
    """Stores the final result of a diagnostic session."""
    
    final_diagnosis: str = Field(description="The final diagnosis rendered")
    ground_truth: str = Field(description="The actual correct diagnosis")
    accuracy_score: float = Field(ge=0, le=5, description="Accuracy score (0-5 scale)")
    accuracy_reasoning: str = Field(description="Reasoning for the accuracy score")
    total_cost: int = Field(description="Total cost of diagnostic workup")
    iterations: int = Field(description="Number of iterations taken")
    conversation_history: str = Field(description="Full conversation log")
    
    @property
    def is_correct(self) -> bool:
        """Check if diagnosis is considered correct (score >= 4)."""
        return self.accuracy_score >= 4.0


# Test cost database - comprehensive medical test costs
TEST_COST_DB = {
    "default": 150,
    "cbc": 50,
    "complete blood count": 50,
    "fbc": 50,
    "chest x-ray": 200,
    "chest xray": 200,
    "mri": 1500,
    "mri brain": 1800,
    "mri neck": 1600,
    "ct scan": 1200,
    "ct chest": 1300,
    "ct abdomen": 1400,
    "biopsy": 800,
    "core biopsy": 900,
    "immunohistochemistry": 400,
    "fish test": 500,
    "fish": 500,
    "ultrasound": 300,
    "ecg": 100,
    "ekg": 100,
    "blood glucose": 30,
    "liver function tests": 80,
    "renal function": 70,
    "toxic alcohol panel": 200,
    "urinalysis": 40,
    "culture": 150,
    "pathology": 600,
}


def get_test_cost(test_name: str) -> int:
    """
    Get the cost for a diagnostic test.
    
    Args:
        test_name: Name of the test (case-insensitive)
        
    Returns:
        Cost of the test in dollars
    """
    test_name_lower = test_name.lower().strip()
    return TEST_COST_DB.get(test_name_lower, TEST_COST_DB["default"])


def create_initial_state(
    initial_vignette: str,
    full_case_details: str,
    ground_truth: str,
    initial_budget: int = 10000,
    max_iterations: int = 10,
    mode: str = "no_budget",
    model_name: str = "gpt-4o-mini",
    physician_visit_cost: int = 300,
    enable_budget_tracking: bool = False,
) -> DiagnosticState:
    """
    Create the initial state for a diagnostic session.
    
    Args:
        initial_vignette: Initial patient presentation
        full_case_details: Complete case information (for gatekeeper)
        ground_truth: Correct diagnosis (for evaluation)
        initial_budget: Starting budget in dollars
        max_iterations: Maximum number of diagnostic iterations
        mode: Operational mode
        model_name: LLM model to use
        physician_visit_cost: Cost per physician consultation
        enable_budget_tracking: Whether to track costs
        
    Returns:
        Initial diagnostic state
    """
    return DiagnosticState(
        messages=[HumanMessage(
            content=f"Initial case presentation:\n{initial_vignette}"
        )],
        initial_vignette=initial_vignette,
        full_case_details=full_case_details,
        ground_truth=ground_truth,
        differential_diagnosis=None,
        current_hypothesis=None,
        final_diagnosis=None,
        cumulative_cost=physician_visit_cost if enable_budget_tracking else 0,
        current_budget=initial_budget,
        iteration=0,
        max_iterations=max_iterations,
        test_results={},
        available_findings={},
        hypothesis_analysis=None,
        test_chooser_analysis=None,
        challenger_analysis=None,
        stewardship_analysis=None,
        checklist_analysis=None,
        stagnation_detected=False,
        ready_for_diagnosis=False,
        current_action=None,
        mode=mode,
        model_name=model_name,
        physician_visit_cost=physician_visit_cost,
        enable_budget_tracking=enable_budget_tracking,
    )