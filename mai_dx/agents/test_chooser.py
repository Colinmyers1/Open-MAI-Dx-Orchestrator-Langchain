"""Dr. Test-Chooser - Diagnostic Test Selection Agent."""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from ..state import DiagnosticState

def test_chooser_agent(state: DiagnosticState, model: BaseChatModel) -> Command:
    """Test chooser agent implementation."""
    # Stub implementation - analyze and recommend tests
    response = model.invoke([HumanMessage(content="Recommend diagnostic tests")])
    return Command(
        goto="consensus",
        update={
            "messages": state["messages"] + [response],
            "test_chooser_analysis": "Test recommendations provided"
        }
    )