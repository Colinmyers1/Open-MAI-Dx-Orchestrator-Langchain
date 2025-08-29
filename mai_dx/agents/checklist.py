"""Dr. Checklist - Quality Control Agent."""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from ..state import DiagnosticState

def checklist_agent(state: DiagnosticState, model: BaseChatModel) -> Command:
    """Checklist agent implementation."""
    response = model.invoke([HumanMessage(content="Quality control review")])
    return Command(
        goto="consensus",
        update={
            "messages": state["messages"] + [response],
            "checklist_analysis": "Quality control completed"
        }
    )