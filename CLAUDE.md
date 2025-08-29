# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Setup
```bash
# Install dependencies (use one of these)
pip install -r requirements.txt
pip install mai-dx  # For using as a package
pip install -e .    # For development mode
```

### Code Quality
```bash
# Linting (configured in pyproject.toml)
ruff check mai_dx/
ruff check mai_dx/ --fix  # Auto-fix issues

# Formatting
black mai_dx/ --line-length 70
```

### Running the Application
```bash
# Run the main demo
python -m mai_dx.main

# Run the example script
python example.py
```

## Architecture Overview

This is an implementation of Microsoft Research's "Sequential Diagnosis with Language Models" paper, built using LangGraph for state-managed multi-agent orchestration. The system coordinates a virtual panel of 8 specialized AI physician agents to perform iterative medical diagnosis with robust state management.

### Core Components

1. **MaiDxOrchestrator** (`mai_dx/main.py`): Main orchestration class using LangGraph StateGraph
   - Manages the diagnostic workflow as a compiled graph
   - Handles state persistence and checkpointing
   - Implements different operational modes (instant, question_only, budgeted, no_budget, ensemble)

2. **StateGraph Architecture**: LangGraph-powered workflow with nodes and edges:
   - **Agent Nodes**: 8 specialized agents as graph nodes with state-aware processing
   - **Control Flow**: Conditional edges based on agent decisions and state
   - **State Management**: Centralized DiagnosticState with automatic persistence

3. **Specialized Agents** (`mai_dx/agents/`):
   - **Dr. Hypothesis**: Maintains differential diagnosis with Bayesian updates
   - **Dr. Test-Chooser**: Selects high-yield diagnostic tests
   - **Dr. Challenger**: Provides critical analysis and alternative perspectives
   - **Dr. Stewardship**: Enforces cost-effective care decisions
   - **Dr. Checklist**: Quality control and validation
   - **Consensus Coordinator**: Synthesizes panel input and makes structured decisions
   - **Gatekeeper**: Clinical information oracle with realistic disclosure
   - **Judge**: Evaluates final diagnosis accuracy

4. **State Management** (`mai_dx/state.py`):
   - **DiagnosticState**: Centralized state with messages, costs, iterations, and agent analyses
   - **DiagnosisResult**: Final results with accuracy scoring
   - **Pydantic Models**: Structured data validation for all components

5. **Tool System** (`mai_dx/tools.py`):
   - **@tool decorators**: Native LangChain tool integration
   - **Command primitives**: State updates and agent handoffs
   - **Function calling**: Structured outputs for consensus and diagnosis

6. **Graph Construction** (`mai_dx/graph.py`):
   - **StateGraph builder**: Defines nodes, edges, and conditional routing
   - **Model integration**: Supports GPT, Claude, Gemini through LangChain
   - **Checkpointing**: Built-in state persistence and resumability

### Key Design Patterns

- **State-Managed Workflow**: LangGraph handles all state transitions and persistence automatically
- **Command-Based Control**: Agents use Command objects for clean handoffs and state updates
- **Tool-Centric Actions**: Native tool calling for structured outputs and test execution
- **Graph-Based Flow**: Declarative workflow definition with conditional routing
- **Multi-Model Support**: Model-agnostic through LangChain's unified interface

### Operational Modes

The system supports 5 distinct modes configured via the `mode` parameter:
- `instant`: Immediate diagnosis from initial presentation
- `question_only`: History-taking without diagnostic tests  
- `budgeted`: Cost-constrained diagnostic workup
- `no_budget`: Full diagnostic capability
- `ensemble`: Multiple independent panels with consensus

### Environment Variables

Required API keys in `.env`:
- `OPENAI_API_KEY`: For GPT models
- `GEMINI_API_KEY`: For Gemini models
- `ANTHROPIC_API_KEY`: For Claude models

Optional:
- `MAIDX_DEBUG=1`: Enable debug logging

## Key Implementation Details

- The system uses the Swarms framework for agent orchestration
- All agents are created with specific token-optimized prompts to reduce latency
- Test costs are stored in a comprehensive database (`test_cost_db`)
- Budget tracking happens in real-time during diagnostic iterations
- The Judge agent uses a 5-point Likert scale for accuracy evaluation
- Conversation history is maintained for full diagnostic trail

## Dependencies

- **langgraph**: State-managed multi-agent workflow orchestration
- **langchain**: Core LLM abstraction and tool integration
- **langchain-openai**: OpenAI model integration (GPT-4, GPT-3.5)
- **langchain-anthropic**: Anthropic model integration (Claude)
- **langchain-google-genai**: Google model integration (Gemini)
- **loguru**: Structured logging with beautiful formatting
- **pydantic**: Data validation and structured models
- **python-dotenv**: Environment variable management