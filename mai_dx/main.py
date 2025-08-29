"""
MAI Diagnostic Orchestrator (MAI-DxO) - LangGraph Implementation

This module provides a complete implementation of the "Sequential Diagnosis with Language Models"
paper using LangGraph for multi-agent orchestration. It simulates a virtual panel of 
physician-agents to perform iterative medical diagnosis with cost-effectiveness optimization.

Based on the paper: "Sequential Diagnosis with Language Models"
(arXiv:2506.22405v1) by Nori et al.

Key Features:
- LangGraph-based multi-agent workflow with specialized roles
- State-managed diagnostic process with built-in checkpointing
- Comprehensive cost tracking and budget management
- Clinical accuracy evaluation with 5-point Likert scale
- Multiple operational modes (instant, question_only, budgeted, no_budget, ensemble)

Example Usage:
    # Standard MAI-DxO usage
    orchestrator = MaiDxOrchestrator(model_name="gpt-4o")
    result = orchestrator.run(initial_case_info, full_case_details, ground_truth)

    # Budget-constrained variant
    budgeted_orchestrator = MaiDxOrchestrator.create_variant("budgeted", budget=5000)

    # Ensemble approach
    ensemble_result = orchestrator.run_ensemble(initial_case_info, full_case_details, ground_truth)
"""

import os
import json
import sys
import time
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from dataclasses import dataclass

from loguru import logger
from dotenv import load_dotenv

from .state import (
    DiagnosticState, DiagnosisResult, AgentRole, 
    create_initial_state, TEST_COST_DB
)
from .graph import compile_diagnostic_graph

load_dotenv()

# Configure Loguru with beautiful formatting
logger.remove()  # Remove default handler

# Console handler with colors  
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
)

# Debug logging if enabled
if os.getenv("MAIDX_DEBUG", "").lower() in ("1", "true", "yes"):
    logger.add(
        "logs/maidx_debug_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="1 day",
        retention="3 days",
    )
    logger.info("🐛 Debug logging enabled - logs will be written to logs/ directory")


class MaiDxOrchestrator:
    """
    MAI Diagnostic Orchestrator using LangGraph for multi-agent coordination.
    
    This class orchestrates a virtual panel of AI agents to perform sequential medical 
    diagnosis, evaluates the final diagnosis, and tracks costs using LangGraph's 
    state management and workflow capabilities.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        max_iterations: int = 10,
        initial_budget: int = 10000,
        mode: str = "no_budget",  # "instant", "question_only", "budgeted", "no_budget", "ensemble"
        physician_visit_cost: int = 300,
        enable_budget_tracking: bool = False,
        enable_checkpointing: bool = True,
    ):
        """
        Initialize the MAI-DxO system with LangGraph architecture.

        Args:
            model_name: Language model to use (supports GPT, Claude, Gemini)
            max_iterations: Maximum diagnostic iterations allowed
            initial_budget: Starting budget in dollars
            mode: Operational mode for different diagnostic approaches
            physician_visit_cost: Cost per physician consultation
            enable_budget_tracking: Whether to track and enforce budget limits
            enable_checkpointing: Whether to enable state persistence
        """
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.initial_budget = initial_budget
        self.mode = mode
        self.physician_visit_cost = physician_visit_cost
        self.enable_budget_tracking = enable_budget_tracking
        self.enable_checkpointing = enable_checkpointing
        
        # Initialize the diagnostic graph
        self.graph = compile_diagnostic_graph(
            model_name=model_name,
            checkpointer=enable_checkpointing
        )
        
        logger.info(
            f"🏥 MAI Diagnostic Orchestrator (LangGraph) initialized successfully in '{mode}' mode with budget ${initial_budget:,}"
        )

    def run(
        self,
        initial_case_info: str,
        full_case_details: str,
        ground_truth: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> DiagnosisResult:
        """
        Run the diagnostic process on a single case.

        Args:
            initial_case_info: Initial patient presentation (what the panel sees first)
            full_case_details: Complete case information (available to Gatekeeper)
            ground_truth: Correct diagnosis for evaluation
            config: Optional configuration for the graph execution

        Returns:
            DiagnosisResult with final diagnosis, accuracy score, and metadata
        """
        logger.info("🔬 Starting diagnostic process...")
        
        start_time = time.time()
        
        # Create initial state
        initial_state = create_initial_state(
            initial_vignette=initial_case_info,
            full_case_details=full_case_details,
            ground_truth=ground_truth,
            initial_budget=self.initial_budget,
            max_iterations=self.max_iterations,
            mode=self.mode,
            model_name=self.model_name,
            physician_visit_cost=self.physician_visit_cost,
            enable_budget_tracking=self.enable_budget_tracking,
        )
        
        # Execute the diagnostic workflow
        try:
            # Provide default config if none provided and checkpointing is enabled
            if config is None and self.enable_checkpointing:
                config = {"configurable": {"thread_id": f"diagnostic_session_{int(start_time)}"}}
            
            final_state = self.graph.invoke(initial_state, config=config)
            logger.info("✅ Diagnostic process completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Diagnostic process failed: {e}")
            raise
        
        # Extract results
        final_diagnosis = final_state.get("final_diagnosis", "No diagnosis reached")
        evaluation = final_state.get("evaluation_result", {})
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Build conversation history
        conversation_history = self._build_conversation_history(final_state["messages"])
        
        # Create result object
        result = DiagnosisResult(
            final_diagnosis=final_diagnosis,
            ground_truth=ground_truth,
            accuracy_score=evaluation.get("accuracy_score", 0.0),
            accuracy_reasoning=evaluation.get("accuracy_reasoning", "No evaluation available"),
            total_cost=final_state.get("cumulative_cost", 0),
            iterations=final_state.get("iteration", 0),
            conversation_history=conversation_history,
        )
        
        # Log results
        logger.info(f"📋 Final Diagnosis: {final_diagnosis}")
        logger.info(f"🎯 Ground Truth: {ground_truth}")
        logger.info(f"📊 Accuracy Score: {result.accuracy_score}/5.0")
        logger.info(f"💰 Total Cost: ${result.total_cost}")
        logger.info(f"🔄 Iterations: {result.iterations}")
        logger.info(f"⏱️  Execution Time: {execution_time:.2f}s")
        
        return result

    def run_streaming(
        self,
        initial_case_info: str,
        full_case_details: str,
        ground_truth: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Run the diagnostic process with streaming updates.

        Args:
            initial_case_info: Initial patient presentation
            full_case_details: Complete case information
            ground_truth: Correct diagnosis for evaluation
            config: Optional configuration for graph execution

        Yields:
            Dictionary updates from each step of the diagnostic process
        """
        logger.info("🔬 Starting streaming diagnostic process...")
        
        # Create initial state
        initial_state = create_initial_state(
            initial_vignette=initial_case_info,
            full_case_details=full_case_details,
            ground_truth=ground_truth,
            initial_budget=self.initial_budget,
            max_iterations=self.max_iterations,
            mode=self.mode,
            model_name=self.model_name,
            physician_visit_cost=self.physician_visit_cost,
            enable_budget_tracking=self.enable_budget_tracking,
        )
        
        # Provide default config if none provided and checkpointing is enabled
        if config is None and self.enable_checkpointing:
            import time
            config = {"configurable": {"thread_id": f"diagnostic_stream_{int(time.time())}"}}
        
        # Stream the execution
        for chunk in self.graph.stream(initial_state, config=config):
            yield chunk

    def run_ensemble(
        self,
        initial_case_info: str,
        full_case_details: str,
        ground_truth: str,
        ensemble_size: int = 3,
        aggregation_method: str = "consensus",
    ) -> DiagnosisResult:
        """
        Run multiple independent diagnostic panels and aggregate results.

        Args:
            initial_case_info: Initial patient presentation
            full_case_details: Complete case information
            ground_truth: Correct diagnosis for evaluation
            ensemble_size: Number of independent panels to run
            aggregation_method: Method for aggregating results ("consensus", "voting", "confidence")

        Returns:
            Aggregated DiagnosisResult from the ensemble
        """
        logger.info(f"🎭 Starting ensemble diagnostic process with {ensemble_size} panels...")
        
        results = []
        for i in range(ensemble_size):
            logger.info(f"Running panel {i+1}/{ensemble_size}...")
            
            # Create independent configuration for each panel
            config = {"configurable": {"thread_id": f"ensemble_{i}"}}
            
            result = self.run(initial_case_info, full_case_details, ground_truth, config)
            results.append(result)
        
        # Aggregate results
        aggregated_result = self._aggregate_ensemble_results(results, aggregation_method, ground_truth)
        
        logger.info(f"🎭 Ensemble diagnosis complete: {aggregated_result.final_diagnosis}")
        return aggregated_result

    @classmethod
    def create_variant(
        cls,
        variant_type: str,
        **kwargs
    ) -> "MaiDxOrchestrator":
        """
        Create a pre-configured variant of the orchestrator.

        Args:
            variant_type: Type of variant ("instant", "budgeted", "research", etc.)
            **kwargs: Additional configuration parameters

        Returns:
            Configured MaiDxOrchestrator instance
        """
        variant_configs = {
            "instant": {
                "mode": "instant",
                "max_iterations": 1,
                "enable_budget_tracking": False,
            },
            "budgeted": {
                "mode": "budgeted", 
                "initial_budget": kwargs.get("budget", 5000),
                "enable_budget_tracking": True,
            },
            "research": {
                "mode": "no_budget",
                "max_iterations": 15,
                "enable_budget_tracking": False,
            },
            "efficient": {
                "mode": "question_only",
                "max_iterations": 8,
                "enable_budget_tracking": True,
                "initial_budget": 3000,
            }
        }
        
        config = variant_configs.get(variant_type, {})
        config.update(kwargs)
        
        logger.info(f"🏭 Creating {variant_type} variant with config: {config}")
        return cls(**config)

    def get_diagnostic_metrics(self, results: List[DiagnosisResult]) -> Dict[str, Any]:
        """
        Calculate diagnostic performance metrics from multiple results.

        Args:
            results: List of DiagnosisResult objects

        Returns:
            Dictionary with aggregated performance metrics
        """
        if not results:
            return {}

        total_cases = len(results)
        correct_cases = sum(1 for r in results if r.is_correct)
        
        accuracy = correct_cases / total_cases
        avg_cost = sum(r.total_cost for r in results) / total_cases
        avg_iterations = sum(r.iterations for r in results) / total_cases
        avg_score = sum(r.accuracy_score for r in results) / total_cases

        return {
            "total_cases": total_cases,
            "correct_diagnoses": correct_cases,
            "diagnostic_accuracy": accuracy,
            "average_cost": avg_cost,
            "average_iterations": avg_iterations,
            "average_accuracy_score": avg_score,
            "cost_per_correct_diagnosis": avg_cost / accuracy if accuracy > 0 else float('inf'),
        }

    def _build_conversation_history(self, messages: List[Dict[str, Any]]) -> str:
        """Build a formatted conversation history from messages."""
        history_parts = []
        
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            if role == "user":
                history_parts.append(f"USER: {content}")
            elif role == "assistant":
                history_parts.append(f"ASSISTANT: {content}")
            elif role == "system":
                history_parts.append(f"SYSTEM: {content}")
            elif role == "tool":
                tool_name = msg.get("name", "unknown_tool")
                history_parts.append(f"TOOL ({tool_name}): {content}")
        
        return "\n".join(history_parts)

    def _aggregate_ensemble_results(
        self, 
        results: List[DiagnosisResult], 
        method: str,
        ground_truth: str
    ) -> DiagnosisResult:
        """Aggregate results from ensemble of diagnostic panels."""
        
        if method == "consensus":
            # Use diagnosis that appears most frequently
            diagnoses = [r.final_diagnosis for r in results]
            diagnosis_counts = {}
            for diag in diagnoses:
                diagnosis_counts[diag] = diagnosis_counts.get(diag, 0) + 1
            
            final_diagnosis = max(diagnosis_counts.keys(), key=diagnosis_counts.get)
            
        elif method == "confidence":
            # Use diagnosis with highest accuracy score
            best_result = max(results, key=lambda r: r.accuracy_score)
            final_diagnosis = best_result.final_diagnosis
            
        else:  # default to consensus
            diagnoses = [r.final_diagnosis for r in results]
            final_diagnosis = max(set(diagnoses), key=diagnoses.count)

        # Calculate aggregated metrics
        avg_cost = sum(r.total_cost for r in results) / len(results)
        avg_iterations = sum(r.iterations for r in results) / len(results)
        avg_score = sum(r.accuracy_score for r in results) / len(results)
        
        # Evaluate final diagnosis
        final_lower = final_diagnosis.lower().strip()
        truth_lower = ground_truth.lower().strip()
        
        if final_lower == truth_lower:
            accuracy_score = 5.0
        elif any(word in final_lower for word in truth_lower.split()):
            accuracy_score = 3.0
        else:
            accuracy_score = 1.0

        # Build ensemble conversation history
        ensemble_history = f"ENSEMBLE RESULTS ({len(results)} panels):\n"
        for i, result in enumerate(results):
            ensemble_history += f"\nPanel {i+1}: {result.final_diagnosis} (Score: {result.accuracy_score})\n"
        ensemble_history += f"\nFinal Consensus: {final_diagnosis}"

        return DiagnosisResult(
            final_diagnosis=final_diagnosis,
            ground_truth=ground_truth,
            accuracy_score=accuracy_score,
            accuracy_reasoning=f"Ensemble aggregation using {method} method",
            total_cost=int(avg_cost),
            iterations=int(avg_iterations),
            conversation_history=ensemble_history,
        )


# Example usage and testing
if __name__ == "__main__":
    # Example case for testing
    test_case = {
        "initial_vignette": """
        Patient: 29-year-old female.
        Chief Complaint: Sore throat and right-sided neck swelling for 7 weeks.
        History: Progressive worsening of right-sided pain and swelling. No fevers, headaches, or GI symptoms.
        Past Medical History: Unremarkable. No smoking or significant alcohol use.
        Physical Exam: Right peritonsillar mass displacing uvula. No other significant findings.
        """,
        
        "full_case_details": """
        Patient: 29-year-old female.
        History: Onset of sore throat 7 weeks prior to admission. Worsening right-sided pain and swelling.
        No fevers, headaches, or gastrointestinal symptoms. Past medical history is unremarkable. No history of smoking or significant alcohol use.
        Physical Exam: Right peritonsillar mass, displacing the uvula. No other significant findings.
        Initial Labs: FBC, clotting studies normal.
        MRI Neck: Showed a large, enhancing mass in the right peritonsillar space.
        Biopsy (H&E): Infiltrative round-cell neoplasm with high nuclear-to-cytoplasmic ratio and frequent mitotic figures.
        Biopsy (Immunohistochemistry for Carcinoma): CD31, D2-40, CD34, ERG, GLUT-1, pan-cytokeratin, CD45, CD20, CD3 all negative. Ki-67: 60% nuclear positivity.
        Biopsy (Immunohistochemistry for Rhabdomyosarcoma): Desmin and MyoD1 diffusely positive. Myogenin multifocally positive.
        Biopsy (FISH): No FOXO1 (13q14) rearrangements detected.
        Final Diagnosis from Pathology: Embryonal rhabdomyosarcoma of the pharynx.
        """,
        
        "ground_truth": "Embryonal rhabdomyosarcoma of the pharynx"
    }
    
    # Initialize orchestrator
    orchestrator = MaiDxOrchestrator(
        model_name="gpt-4o-mini",
        mode="no_budget",
        max_iterations=8,
    )
    
    # Run diagnostic process
    result = orchestrator.run(
        initial_case_info=test_case["initial_vignette"],
        full_case_details=test_case["full_case_details"],
        ground_truth=test_case["ground_truth"]
    )
    
    print(f"\n📋 Diagnostic Result:")
    print(f"Final Diagnosis: {result.final_diagnosis}")
    print(f"Ground Truth: {result.ground_truth}")
    print(f"Accuracy Score: {result.accuracy_score}/5.0")
    print(f"Total Cost: ${result.total_cost}")
    print(f"Iterations: {result.iterations}")
    print(f"Correct: {'✅' if result.is_correct else '❌'}")