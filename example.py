from mai_dx.main import MaiDxOrchestrator
from loguru import logger

if __name__ == "__main__":
    # Example case inspired by the paper's Figure 1
    initial_info = (
        "A 29-year-old woman was admitted to the hospital because of sore throat and peritonsillar swelling "
        "and bleeding. Symptoms did not abate with antimicrobial therapy."
    )

    full_case = """
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
    """

    ground_truth = "Embryonal rhabdomyosarcoma of the pharynx"

    # --- Demonstrate Different MAI-DxO Variants ---
    try:
        print("\n" + "=" * 80)
        print(
            "    MAI DIAGNOSTIC ORCHESTRATOR (MAI-DxO) - SEQUENTIAL DIAGNOSIS BENCHMARK"
        )
        print(
            "                    Implementation based on the NEJM Research Paper"
        )
        print("=" * 80)

        # Test different variants as described in the paper
        # Test only one variant to respect Gemini free tier rate limits (10 requests/minute)
        variants_to_test = [
            (
                "no_budget",
                "Standard MAI-DxO with no budget constraints",
            ),
        ]

        results = {}

        for variant_name, description in variants_to_test:
            print(f"\n{'='*60}")
            print(f"Testing Variant: {variant_name.upper()}")
            print(f"Description: {description}")
            print("=" * 60)

            # Create the variant
            # Using Gemini 2.5 Flash with upgraded credits
            orchestrator = MaiDxOrchestrator.create_variant(
                variant_name,
                model_name="gemini-2.5-flash",  # Updated to latest Gemini model
                max_iterations=3,  # Increased iterations with better credits
            )

            # Run the diagnostic process
            result = orchestrator.run(
                initial_case_info=initial_info,
                full_case_details=full_case,
                ground_truth=ground_truth,
            )

            results[variant_name] = result

            # Display results
            print(f"\n🚀 Final Diagnosis: {result.final_diagnosis}")
            print(f"🎯 Ground Truth: {result.ground_truth}")
            print(f"⭐ Accuracy Score: {result.accuracy_score}/5.0")
            print(f"   Reasoning: {result.accuracy_reasoning}")
            print(f"💰 Total Cost: ${result.total_cost:,}")
            print(f"🔄 Iterations: {result.iterations}")
            print(f"⏱️  Mode: {orchestrator.mode}")

        # Skip ensemble testing to respect rate limits
        print(f"\n{'='*60}")
        print("Skipping ensemble testing to respect Gemini free tier rate limits")
        print("=" * 60)

        # --- Summary Comparison ---
        print(f"\n{'='*80}")
        print("                           RESULTS SUMMARY")
        print("=" * 80)
        print(
            f"{'Variant':<15} {'Diagnosis Match':<15} {'Score':<8} {'Cost':<12} {'Iterations':<12}"
        )
        print("-" * 80)

        for variant_name, result in results.items():
            match_status = (
                "✓ Match"
                if result.accuracy_score >= 4.0
                else "✗ No Match"
            )
            print(
                f"{variant_name:<15} {match_status:<15} {result.accuracy_score:<8.1f} ${result.total_cost:<11,} {result.iterations:<12}"
            )

        print(f"\n{'='*80}")
        print(
            "Implementation successfully demonstrates the MAI-DxO framework"
        )
        print(
            "as described in 'Sequential Diagnosis with Language Models' paper"
        )
        print("=" * 80)

    except Exception as e:
        logger.exception(
            f"An error occurred during the diagnostic session: {e}"
        )
        print(f"\n❌ Error occurred: {e}")
        print("Please check your model configuration and API keys.")
