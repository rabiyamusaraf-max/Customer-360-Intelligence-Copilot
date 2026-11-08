"""Run all project stages in sequence. Use this after installing requirements."""
from src.data_pipeline import audit,clean_data,validate
from src.customer_models import rfm_and_segments,build_lapse_model
from src.triage import ROOT, run_pipeline,FEW
from src.policy_rag import PolicyRAG,evaluate
def main():
    print("\n=== PART A: AUDIT / CLEAN / VALIDATE ===")
    audit(); clean_data(); validate()
    print("\n=== PART B: SEGMENTS ===")
    rfm,diagnostics,profile,pca=rfm_and_segments()
    print("K-means diagnostics:\n",diagnostics.to_string(index=False))
    print("Segment profile:\n",profile.to_string(index=False)); print("PCA explained variance:",pca)
    print("Lapse model summary:",build_lapse_model())
    print("\n=== PART C: TRIAGE ===")
    triage,summary=run_pipeline(prompt=FEW,token_budget=100000)
    print(summary); print(triage.head().to_string(index=False))
    print("\n=== PART D: POLICY RAG, REFUND AGENT, SAFETY AND RESUME ===")
    # Import after Part A has created the SQLite database.
    from run_part_d import main as run_part_d
    run_part_d()
    
    print("\n=== COMPLETE PROJECT OUTPUTS ===")
    expected = [
        "outputs/customer_segments.csv",
        "outputs/lapse_model.joblib",
        "outputs/retention_list.csv",
        "outputs/triage_results.csv",
        "outputs/triage_budget_demo.csv",
        "outputs/policy_eval_results.csv",
        "outputs/policy_eval_zero_overlap.csv",
        "outputs/refund_agent_eval.csv",
        "outputs/agent_runs.jsonl",
        "outputs/refund_ledger.jsonl",
        "outputs/part_d_summary.json",
        "data/instamart.db",
    ]
    missing = []
    for relative in expected:
        exists = (ROOT / relative).exists() 
        print(f"{'OK' if exists else 'MISSING'}  {relative}")
        if not exists:
            missing.append(relative)
        if missing:
            raise RuntimeError("Missing expected artifacts: " + ", ".join(missing))
        print("\nPROJECT PIPELINE FINISHED SUCCESSFULLY.")
    
    
if __name__ == "__main__":
        main()