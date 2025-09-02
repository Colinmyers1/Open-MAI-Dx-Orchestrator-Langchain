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
from langsmith import traceable
from langchain_core.tracers.context import tracing_enabled
from langchain_core.messages import BaseMessage

from .state import (
    DiagnosticState, DiagnosisResult, AgentRole, 
    create_initial_state
)
from .graph_unified import compile_unified_diagnostic_graph

load_dotenv()

# Configure LangSmith tracing
def setup_langsmith_tracing():
    """Configure LangSmith tracing with best practices."""
    langchain_tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower()
    langchain_api_key = os.getenv("LANGCHAIN_API_KEY", "")
    
    if langchain_tracing == "true" and langchain_api_key:
        logger.info("🔍 LangSmith tracing enabled - monitoring diagnostic workflows")
        return True
    elif langchain_tracing == "true" and not langchain_api_key:
        logger.warning("⚠️  LangSmith tracing enabled but LANGCHAIN_API_KEY not set")
        return False
    else:
        logger.info("📝 LangSmith tracing disabled")
        return False

# Initialize tracing
LANGSMITH_ENABLED = setup_langsmith_tracing()

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
        model_name: str = "gemini-2.0-flash-001",
        max_iterations: int = 10,
        initial_budget: int = 10000,
        mode: str = "no_budget",  # "instant", "question_only", "budgeted", "no_budget", "ensemble"
        physician_visit_cost: int = 300,
        enable_budget_tracking: bool = False,
        enable_checkpointing: bool = True,
        evaluation_mode: bool = False,  # True for testing known cases, False for real cases
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
            evaluation_mode: True for testing against known cases, False for real cases
        """
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.initial_budget = initial_budget
        self.mode = mode
        self.physician_visit_cost = physician_visit_cost
        self.enable_budget_tracking = enable_budget_tracking
        self.enable_checkpointing = enable_checkpointing
        self.evaluation_mode = evaluation_mode
        
        # Initialize the unified diagnostic graph (paper-aligned)
        self.graph = compile_unified_diagnostic_graph(
            model_name=model_name
        )
        
        mode_desc = f"'{mode}' mode ({'evaluation' if evaluation_mode else 'new case'})"
        logger.info(
            f"🏥 MAI Diagnostic Orchestrator (LangGraph) initialized successfully in {mode_desc} with budget ${initial_budget:,}"
        )

    @traceable(name="mai_dx_evaluation_workflow", run_type="chain")
    def run_evaluation(
        self,
        initial_case_info: str,
        full_case_details: str,
        ground_truth: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> DiagnosisResult:
        """
        Run diagnostic evaluation on a known case (uses Gatekeeper & Judge).
        
        Args:
            initial_case_info: Initial patient presentation
            full_case_details: Complete case information for Gatekeeper
            ground_truth: Correct diagnosis for Judge evaluation
            config: Optional configuration for graph execution
            
        Returns:
            DiagnosisResult with evaluation against ground truth
        """
        return self._run_diagnostic_workflow(
            initial_case_info=initial_case_info,
            full_case_details=full_case_details,
            ground_truth=ground_truth,
            evaluation_mode=True,
            config=config
        )
    
    @traceable(name="mai_dx_new_case_workflow", run_type="chain")
    def run_new_case(
        self,
        initial_case_info: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> DiagnosisResult:
        """
        Run diagnostic process on a new unknown case (real clinical work).
        
        Args:
            initial_case_info: Initial patient presentation
            config: Optional configuration for graph execution
            
        Returns:
            DiagnosisResult with final diagnosis (no evaluation)
        """
        return self._run_diagnostic_workflow(
            initial_case_info=initial_case_info,
            full_case_details="",
            ground_truth="",
            evaluation_mode=False,
            config=config
        )
    
    @traceable(name="mai_dx_diagnostic_workflow", run_type="chain")  
    def run(
        self,
        initial_case_info: str,
        full_case_details: str = "",
        ground_truth: str = "",
        config: Optional[Dict[str, Any]] = None,
    ) -> DiagnosisResult:
        """
        Run the diagnostic process (backwards compatible method).

        Args:
            initial_case_info: Initial patient presentation
            full_case_details: Complete case information (for evaluation mode)
            ground_truth: Correct diagnosis (for evaluation mode)
            config: Optional configuration for graph execution

        Returns:
            DiagnosisResult with final diagnosis and metadata
        """
        # Determine mode based on whether we have ground truth
        evaluation_mode = bool(ground_truth) or bool(full_case_details)
        
        return self._run_diagnostic_workflow(
            initial_case_info=initial_case_info,
            full_case_details=full_case_details,
            ground_truth=ground_truth,
            evaluation_mode=evaluation_mode,
            config=config
        )
    
    def _run_diagnostic_workflow(
        self,
        initial_case_info: str,
        full_case_details: str,
        ground_truth: str,
        evaluation_mode: bool,
        config: Optional[Dict[str, Any]] = None,
    ) -> DiagnosisResult:
        """
        Internal method to run the diagnostic workflow.
        
        Args:
            initial_case_info: Initial patient presentation
            full_case_details: Complete case information
            ground_truth: Correct diagnosis
            evaluation_mode: True for evaluation, False for new cases
            config: Optional configuration
            
        Returns:
            DiagnosisResult with diagnosis and metadata
        """
        workflow_type = "evaluation" if evaluation_mode else "new case"
        logger.info(f"🔬 Starting {workflow_type} diagnostic process...")
        
        start_time = time.time()
        
        # Add trace metadata for better debugging
        trace_metadata = {
            "model_name": self.model_name,
            "mode": self.mode,
            "evaluation_mode": evaluation_mode,
            "max_iterations": self.max_iterations,
            "initial_budget": self.initial_budget,
            "case_complexity": "complex" if len(initial_case_info) > 1000 else "standard",
            "enable_budget_tracking": self.enable_budget_tracking,
        }
        
        # Create initial state - only provide brief case prompt to agents
        initial_state = create_initial_state(
            initial_case_prompt=initial_case_info,  # Brief prompt only
            full_case_details=full_case_details,  # Gatekeeper access only
            ground_truth=ground_truth,
            initial_budget=self.initial_budget,
            mode=self.mode,
            model_name=self.model_name,
            physician_visit_cost=self.physician_visit_cost,
            enable_budget_tracking=self.enable_budget_tracking,
            evaluation_mode=evaluation_mode,
            max_iterations=self.max_iterations,
        )
        
        # Execute the diagnostic workflow
        try:
            # Provide default config if none provided and checkpointing is enabled
            if config is None and self.enable_checkpointing:
                config = {"configurable": {"thread_id": f"diagnostic_session_{int(start_time)}"}}
            elif config is None:
                config = {}
            
            # Add recursion limit to prevent infinite loops (reduced for testing)
            if "recursion_limit" not in config:
                config["recursion_limit"] = 50  # Reduced limit for safer testing - prevents infinite loops
            
            # Add tracing metadata to config
            if LANGSMITH_ENABLED:
                config.setdefault("metadata", {}).update(trace_metadata)
                config["run_name"] = f"MAI-DX Diagnostic Session ({self.mode})"
            
            final_state = self.graph.invoke(initial_state, config=config)
            logger.info("✅ Diagnostic process completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Diagnostic process failed: {e}")
            raise
        
        # Extract results - ensure final_diagnosis is always a string
        final_diagnosis = final_state.get("final_diagnosis") or "No diagnosis reached - maximum iterations exceeded"
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

    @traceable(name="mai_dx_streaming_workflow", run_type="chain")
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
        
        # Create initial state - only provide brief case prompt to agents  
        initial_state = create_initial_state(
            initial_case_prompt=initial_case_info,  # Brief prompt only
            full_case_details=full_case_details,  # Gatekeeper access only
            ground_truth=ground_truth,
            initial_budget=self.initial_budget,
            mode=self.mode,
            model_name=self.model_name,
            physician_visit_cost=self.physician_visit_cost,
            enable_budget_tracking=self.enable_budget_tracking,
            max_iterations=self.max_iterations,
        )
        
        # Provide default config if none provided and checkpointing is enabled
        if config is None and self.enable_checkpointing:
            import time
            config = {"configurable": {"thread_id": f"diagnostic_stream_{int(time.time())}"}}
        
        # Stream the execution
        for chunk in self.graph.stream(initial_state, config=config):
            yield chunk

    @traceable(name="mai_dx_ensemble_workflow", run_type="chain")
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

    def _build_conversation_history(self, messages: List[BaseMessage]) -> str:
        """Build a formatted conversation history from LangChain messages."""
        history_parts = []
        
        for msg in messages:
            # Handle different LangChain message types
            if hasattr(msg, '__class__'):
                msg_type = msg.__class__.__name__.lower()
                content = getattr(msg, 'content', '')
                
                if 'human' in msg_type or 'user' in msg_type:
                    history_parts.append(f"USER: {content}")
                elif 'ai' in msg_type or 'assistant' in msg_type:
                    history_parts.append(f"ASSISTANT: {content}")
                elif 'system' in msg_type:
                    history_parts.append(f"SYSTEM: {content}")
                elif 'tool' in msg_type:
                    tool_name = getattr(msg, 'name', 'unknown_tool')
                    history_parts.append(f"TOOL ({tool_name}): {content}")
                else:
                    # Fallback for unknown message types
                    history_parts.append(f"{msg_type.upper()}: {content}")
        
        return "\n".join(history_parts)

    def _aggregate_ensemble_results(
        self, 
        results: List[DiagnosisResult], 
        method: str,
        ground_truth: str
    ) -> DiagnosisResult:
        """
        Aggregate results from ensemble of diagnostic panels using sophisticated methods.
        
        Args:
            results: List of diagnosis results from independent panels
            method: Aggregation method ("voting", "confidence", "weighted", "bayesian")
            ground_truth: Correct diagnosis for evaluation
            
        Returns:
            Aggregated diagnosis result
        """
        if not results:
            raise ValueError("No results to aggregate")
        
        if method == "voting":
            # Simple majority voting
            diagnoses = [r.final_diagnosis.lower().strip() for r in results]
            diagnosis_counts = {}
            for diag in diagnoses:
                diagnosis_counts[diag] = diagnosis_counts.get(diag, 0) + 1
            
            final_diagnosis = max(diagnosis_counts.keys(), key=diagnosis_counts.get)
            reasoning = f"Majority vote: {diagnosis_counts[final_diagnosis]}/{len(results)} panels"
            
        elif method == "confidence":
            # Use diagnosis from most confident panel
            best_result = max(results, key=lambda r: r.accuracy_score)
            final_diagnosis = best_result.final_diagnosis.lower().strip()
            reasoning = f"Highest confidence panel (score: {best_result.accuracy_score})"
            
        elif method == "weighted":
            # Weight votes by confidence/accuracy scores
            diagnosis_weights = {}
            for result in results:
                diag = result.final_diagnosis.lower().strip()
                weight = result.accuracy_score if result.accuracy_score > 0 else 1.0
                diagnosis_weights[diag] = diagnosis_weights.get(diag, 0) + weight
            
            final_diagnosis = max(diagnosis_weights.keys(), key=diagnosis_weights.get)
            reasoning = f"Confidence-weighted voting (weight: {diagnosis_weights[final_diagnosis]:.2f})"
            
        elif method == "bayesian":
            # Bayesian model averaging
            diagnosis_probs = {}
            total_confidence = sum(max(r.accuracy_score, 0.1) for r in results)
            
            for result in results:
                diag = result.final_diagnosis.lower().strip()
                # Use accuracy score as confidence weight
                confidence = max(result.accuracy_score, 0.1)
                prob = confidence / total_confidence
                diagnosis_probs[diag] = diagnosis_probs.get(diag, 0) + prob
            
            final_diagnosis = max(diagnosis_probs.keys(), key=diagnosis_probs.get)
            reasoning = f"Bayesian averaging (posterior: {diagnosis_probs[final_diagnosis]:.3f})"
            
        else:  # default to voting
            diagnoses = [r.final_diagnosis.lower().strip() for r in results]
            final_diagnosis = max(set(diagnoses), key=diagnoses.count)
            reasoning = "Default consensus voting"

        # Calculate ensemble metrics
        total_cost = sum(r.total_cost for r in results)  # Sum costs (all tests were run)
        avg_iterations = sum(r.iterations for r in results) / len(results)
        ensemble_confidence = self._calculate_ensemble_confidence(results, final_diagnosis)
        
        # Evaluate final diagnosis against ground truth
        accuracy_score = self._evaluate_diagnosis_accuracy(final_diagnosis, ground_truth)

        # Build detailed ensemble conversation history
        ensemble_history = self._build_ensemble_history(results, final_diagnosis, method, reasoning)

        return DiagnosisResult(
            final_diagnosis=final_diagnosis.title(),  # Restore proper capitalization
            ground_truth=ground_truth,
            accuracy_score=accuracy_score,
            accuracy_reasoning=f"Ensemble {method}: {reasoning}. Agreement: {ensemble_confidence:.1%}",
            total_cost=total_cost,
            iterations=int(avg_iterations),
            conversation_history=ensemble_history,
        )
    
    def _calculate_ensemble_confidence(self, results: List[DiagnosisResult], final_diagnosis: str) -> float:
        """Calculate confidence in the ensemble decision."""
        if not results:
            return 0.0
        
        # Count how many panels agreed with the final diagnosis
        agreements = sum(1 for r in results if r.final_diagnosis.lower().strip() == final_diagnosis.lower().strip())
        return agreements / len(results)
    
    def _evaluate_diagnosis_accuracy(self, diagnosis: str, ground_truth: str) -> float:
        """Evaluate diagnosis accuracy using semantic matching."""
        diag_lower = diagnosis.lower().strip()
        truth_lower = ground_truth.lower().strip()
        
        # Exact match
        if diag_lower == truth_lower:
            return 5.0
        
        # High similarity (most words match)
        diag_words = set(word for word in diag_lower.split() if len(word) > 2)
        truth_words = set(word for word in truth_lower.split() if len(word) > 2)
        
        if diag_words and truth_words:
            overlap = len(diag_words.intersection(truth_words))
            union = len(diag_words.union(truth_words))
            similarity = overlap / union if union > 0 else 0
            
            if similarity >= 0.8:
                return 4.5
            elif similarity >= 0.6:
                return 4.0
            elif similarity >= 0.4:
                return 3.0
            elif similarity >= 0.2:
                return 2.0
        
        # Partial word matching
        if any(word in diag_lower for word in truth_words if len(word) > 3):
            return 2.0
        
        # No meaningful match
        return 1.0
    
    def _build_ensemble_history(
        self, 
        results: List[DiagnosisResult], 
        final_diagnosis: str, 
        method: str,
        reasoning: str
    ) -> str:
        """Build detailed ensemble conversation history."""
        history_parts = [
            f"=== ENSEMBLE DIAGNOSTIC RESULTS ===",
            f"Method: {method.title()}",
            f"Panels: {len(results)}",
            f"Final Diagnosis: {final_diagnosis}",
            f"Reasoning: {reasoning}",
            "",
            "Individual Panel Results:"
        ]
        
        for i, result in enumerate(results, 1):
            cost_per_test = result.total_cost / max(result.iterations, 1)
            history_parts.append(
                f"Panel {i}: {result.final_diagnosis} "
                f"(Confidence: {result.accuracy_score:.1f}/5, "
                f"Cost: ${result.total_cost}, "
                f"Tests: ~{cost_per_test:.0f}/test)"
            )
        
        # Add ensemble statistics
        total_cost = sum(r.total_cost for r in results)
        avg_confidence = sum(r.accuracy_score for r in results) / len(results)
        agreement = self._calculate_ensemble_confidence(results, final_diagnosis)
        
        history_parts.extend([
            "",
            f"=== ENSEMBLE STATISTICS ===",
            f"Total Cost: ${total_cost}",
            f"Average Panel Confidence: {avg_confidence:.1f}/5",
            f"Panel Agreement: {agreement:.1%}",
            f"Cost Efficiency: ${total_cost/len(results):.0f} per independent diagnosis"
        ])
        
        return "\n".join(history_parts)


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
    
    # Example 1: Evaluation Mode (testing against known case)
    print("🧪 EVALUATION MODE EXAMPLE")
    print("=" * 40)
    
    eval_orchestrator = MaiDxOrchestrator(
        model_name="gemini-2.0-flash-001",
        mode="no_budget",
        max_iterations=8,
        evaluation_mode=True,
    )
    
    # Run evaluation on known case (uses Gatekeeper + Judge)
    eval_result = eval_orchestrator.run_evaluation(
        initial_case_info=test_case["initial_vignette"],
        full_case_details=test_case["full_case_details"],
        ground_truth=test_case["ground_truth"]
    )
    
    print(f"\n📋 Evaluation Results:")
    print(f"Final Diagnosis: {eval_result.final_diagnosis}")
    print(f"Ground Truth: {eval_result.ground_truth}")
    print(f"Accuracy Score: {eval_result.accuracy_score}/5.0")
    print(f"Total Cost: ${eval_result.total_cost}")
    print(f"Iterations: {eval_result.iterations}")
    print(f"Correct: {'✅' if eval_result.is_correct else '❌'}")
    
    # Example 2: New Case Mode (real diagnostic work)
    print("\n\n🏥 NEW CASE MODE EXAMPLE")
    print("=" * 40)
    
    new_case_orchestrator = MaiDxOrchestrator(
        model_name="gemini-2.0-flash-001",
        mode="no_budget",
        max_iterations=6,
        evaluation_mode=False,
    )
    
    # Run diagnostic on new case (no ground truth, uses user interaction)
    new_case_result = new_case_orchestrator.run_new_case(
        initial_case_info=test_case["initial_vignette"]
    )
    
    print(f"\n📋 New Case Results:")
    print(f"Final Diagnosis: {new_case_result.final_diagnosis}")
    print(f"Accuracy Score: {new_case_result.accuracy_score} (no evaluation)")
    print(f"Total Cost: ${new_case_result.total_cost}")
    print(f"Iterations: {new_case_result.iterations}")
    
    # Show the difference
    result = eval_result  # For backwards compatibility
    
    print(f"\n📋 Diagnostic Result:")
    print(f"Final Diagnosis: {result.final_diagnosis}")
    print(f"Ground Truth: {result.ground_truth}")
    print(f"Accuracy Score: {result.accuracy_score}/5.0")
    print(f"Total Cost: ${result.total_cost}")
    print(f"Iterations: {result.iterations}")
    print(f"Correct: {'✅' if result.is_correct else '❌'}")