"""Dr. Stewardship - Cost Analysis Agent."""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from ..state import DiagnosticState

def stewardship_agent(state: DiagnosticState, model: BaseChatModel) -> Command:
    """Stewardship agent implementation."""
    response = model.invoke([HumanMessage(content="Analyze cost-effectiveness")])
    return Command(
        goto="consensus",
        update={
            "messages": state["messages"] + [response],
            "stewardship_analysis": "Cost analysis provided"
        }
    )