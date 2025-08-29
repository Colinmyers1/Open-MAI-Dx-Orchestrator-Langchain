"""Dr. Challenger - Critical Analysis Agent."""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from ..state import DiagnosticState

def challenger_agent(state: DiagnosticState, model: BaseChatModel) -> Command:
    """Challenger agent implementation."""
    response = model.invoke([HumanMessage(content="Provide critical analysis")])
    return Command(
        goto="consensus",
        update={
            "messages": state["messages"] + [response],
            "challenger_analysis": "Critical analysis provided"
        }
    )