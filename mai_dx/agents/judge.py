"""Judge - Diagnosis Evaluation Agent."""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.types import Command
from ..state import DiagnosticState
from ..tools import evaluate_diagnosis_accuracy

def judge_agent(state: DiagnosticState, model: BaseChatModel) -> Command:
    """Judge agent implementation."""
    final_diagnosis = state.get("final_diagnosis", "No diagnosis provided")
    ground_truth = state["ground_truth"]
    
    # Simple evaluation logic (avoiding tool call complexity for now)
    final_lower = final_diagnosis.lower()
    truth_lower = ground_truth.lower()
    
    if final_lower == truth_lower:
        accuracy_score = 5.0
        reasoning = "Exact match"
    elif any(word in final_lower for word in truth_lower.split() if len(word) > 3):
        accuracy_score = 3.0
        reasoning = "Partial match"
    else:
        accuracy_score = 1.0
        reasoning = "No significant match"
        
    evaluation = {
        "accuracy_score": accuracy_score,
        "accuracy_reasoning": reasoning
    }
    
    response = AIMessage(
        content=f"Diagnosis evaluation complete. Score: {evaluation['accuracy_score']}/5.0"
    )
    
    return Command(
        goto="__end__",
        update={
            "messages": state["messages"] + [response],
            "evaluation_result": evaluation
        }
    )