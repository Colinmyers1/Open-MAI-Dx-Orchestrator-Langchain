#!/usr/bin/env python3
"""
Simple test of the unified diagnostic agent to debug issues.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from mai_dx.main import MaiDxOrchestrator

# Setup simple logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

def test_simple_case():
    """Test with a very simple case."""
    
    logger.info("🧪 Testing unified diagnostic agent with simple case")
    
    # Check environment
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("❌ GEMINI_API_KEY not found")
        return False
    
    try:
        # Initialize orchestrator
        orchestrator = MaiDxOrchestrator(
            model_name="gemini-2.0-flash-001", 
            mode="no_budget",
            max_iterations=3,  # Limit iterations
            initial_budget=1000,
            enable_budget_tracking=False,
            evaluation_mode=True
        )
        
        # Simple test case
        initial_case = "A 30-year-old patient presents with chest pain."
        case_details = "Patient has sharp chest pain, worse with breathing."
        ground_truth = "Pleuritic chest pain"
        
        logger.info("🏃 Running diagnosis...")
        
        # Run diagnosis
        result = orchestrator.run(
            initial_case_info=initial_case,
            full_case_details=case_details,
            ground_truth=ground_truth
        )
        
        logger.info(f"✅ Test completed successfully!")
        logger.info(f"Final diagnosis: {result.final_diagnosis}")
        logger.info(f"Total cost: ${result.total_cost}")
        logger.info(f"Iterations: {result.iterations}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_simple_case()
    sys.exit(0 if success else 1)