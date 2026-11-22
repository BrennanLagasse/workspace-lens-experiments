"""
Compute the correctness of a base model on various benchmark tasks in advance of evaluation
"""

import pandas as pd
from lens_evals.readout_evals.readout_eval_items import MODEL_CORRECTNESS_CSV, load_readout_eval_items
from lens_evals.readout_evals.readout_evals import ReadoutEvalRunner
from workspace_lens import get_hf_model

import time
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen3.5-9B", help="Huggingface model name of model to evaluate", type=str)
    args = p.parse_args()

    print(f"\nStarting evaluation on {args.model_name}\n")

    t1 = time.time()

    model = get_hf_model(args.model_name)  # pass device_map= etc. as for jpp_cli
    runner = ReadoutEvalRunner(model, {}, layers=[0], max_seq_len=512)  # layers is unused for grading
    new_df = runner.grade_model_correctness(load_readout_eval_items(), hf_model_name=args.model_name)

    existing_df = pd.read_csv(MODEL_CORRECTNESS_CSV).drop(columns=[args.model_name], errors="ignore")
    existing_df.merge(new_df, on=["eval", "item"], how="outer").to_csv(MODEL_CORRECTNESS_CSV, index=False)

    t2 = time.time()

    print(f"\nCompleted evaluation in {t2 - t1}s")
    print("Results written to data/jlens/evaluations/model_correctness.csv\n")

if __name__ == "__main__":
    main()
