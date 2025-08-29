# MAI Diagnostic Orchestrator (MAI-DxO)

> **An open-source implementation of Microsoft Research's "Sequential Diagnosis with Language Models" paper, built with LangGraph for state-managed multi-agent orchestration.**

MAI-DxO (MAI Diagnostic Orchestrator) is a sophisticated AI-powered diagnostic system that simulates a virtual panel of physician-agents to perform iterative medical diagnosis with cost-effectiveness optimization. This implementation faithfully reproduces the methodology described in the Microsoft Research paper while leveraging LangGraph's powerful state management and workflow orchestration capabilities.

[![Paper](https://img.shields.io/badge/Paper-arXiv:2506.22405-red.svg)](https://arxiv.org/abs/2506.22405)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)

## ✨ Key Features

- **8 AI Physician Agents**: Specialized roles orchestrated through LangGraph state management.
- **5 Operational Modes**: Instant, question-only, budgeted, no-budget, and ensemble modes.
- **State-Managed Workflow**: LangGraph handles all state transitions and persistence automatically.
- **Cost Tracking**: Real-time budget monitoring with costs for 25+ medical tests.
- **Clinical Evaluation**: 5-point accuracy scoring with detailed feedback.
- **Model Agnostic**: Works with GPT, Gemini, Claude, and other leading LLMs through LangChain.
- **Graph-Based Architecture**: Declarative workflow definition with conditional routing and checkpointing.

## 🚀 Quick Start

### 1. Installation

Clone the repository and install the requirements:

```bash
git clone https://github.com/colinmyers/Open-MAI-Dx-Orchestrator-Langchain.git
cd Open-MAI-Dx-Orchestrator-Langchain
pip install -r requirements.txt
```

For development mode:

```bash
pip install -e .
```

### 2. Environment Setup

Create a `.env` file in your project root and add your API keys:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API keys:
# GEMINI_API_KEY="your_gemini_api_key_here"
# OPENAI_API_KEY="your_openai_api_key_here" 
# ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

### 3. Basic Usage

```python
from mai_dx import MaiDxOrchestrator

# Create the orchestrator (defaults to a capable model)
orchestrator = MaiDxOrchestrator()

# Run a diagnosis
result = orchestrator.run(
    initial_case_info="29-year-old woman with sore throat and peritonsillar swelling...",
    full_case_details="Patient: 29-year-old female. History: Onset of sore throat...",
    ground_truth_diagnosis="Embryonal rhabdomyosarcoma of the pharynx"
)

# Print the results
print(f"Final Diagnosis: {result.final_diagnosis}")
print(f"Accuracy: {result.accuracy_score}/5.0")
print(f"Total Cost: ${result.total_cost:,.2f}")
```

## ⚙️ Advanced Usage & Configuration

Customize the orchestrator's model, budget, and operational mode.

```python
from mai_dx import MaiDxOrchestrator

# Configure with a specific model and budget
orchestrator = MaiDxOrchestrator(
    model_name="gemini/gemini-2.5-flash",  # or "gpt-4", "claude-3-5-sonnet"
    max_iterations=10,
    initial_budget=3000,
    mode="budgeted"  # Other modes: "instant", "question_only", "no_budget"
)

# Run the diagnosis
# ...
```

## 🏥 How It Works: The Virtual Physician Panel

MAI-DxO employs a LangGraph-orchestrated multi-agent system where each agent operates as a graph node with state-aware processing:

- **🧠 Dr. Hypothesis**: Maintains the differential diagnosis.
- **🔬 Dr. Test-Chooser**: Selects the most cost-effective diagnostic tests.
- **🤔 Dr. Challenger**: Prevents cognitive biases and diagnostic errors.
- **💰 Dr. Stewardship**: Ensures cost-effective care.
- **✅ Dr. Checklist**: Performs quality control checks.
- **🤝 Consensus Coordinator**: Synthesizes panel decisions.
- **🔑 Gatekeeper**: Acts as the clinical information oracle.
- **⚖️ Judge**: Evaluates the final diagnostic accuracy.


## Documentation

Learn more about this repository [with the docs](DOCS.md)

## 🤝 Contributing

We welcome contributions! Please feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this work in your research, please cite both the original paper and this software implementation.

```bibtex
@misc{nori2025sequentialdiagnosislanguagemodels,
    title={Sequential Diagnosis with Language Models}, 
    author={Harsha Nori and Mayank Daswani and Christopher Kelly and Scott Lundberg and Marco Tulio Ribeiro and Marc Wilson and Xiaoxuan Liu and Viknesh Sounderajah and Jonathan Carlson and Matthew P Lungren and Bay Gross and Peter Hames and Mustafa Suleyman and Dominic King and Eric Horvitz},
    year={2025},
    eprint={2506.22405},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2506.22405}, 
}

@software{mai_dx_orchestrator,
    title={Open-MAI-Dx-Orchestrator-Langchain: A LangGraph Implementation of Sequential Diagnosis with Language Models},
    author={Colin Myers},
    year={2025},
    url={https://github.com/colinmyers/Open-MAI-Dx-Orchestrator-Langchain.git}
}
```

## 🔗 Related Work

- [Original Paper](https://arxiv.org/abs/2506.22405) - Sequential Diagnosis with Language Models
- [LangGraph](https://github.com/langchain-ai/langgraph) - State-managed multi-agent workflow orchestration
- [LangChain](https://github.com/langchain-ai/langchain) - LLM application framework
- [Microsoft Research](https://www.microsoft.com/en-us/research/) - Original research institution

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/colinmyers/Open-MAI-Dx-Orchestrator-Langchain/issues)
- **Discussions**: [GitHub Discussions](https://github.com/colinmyers/Open-MAI-Dx-Orchestrator-Langchain/discussions)
- **Documentation**: See CLAUDE.md for implementation details

---

<p align="center">
  <strong>Built with LangGraph for advancing AI-powered medical diagnosis</strong>
</p>