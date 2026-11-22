"""
Review the percent accuracy across base models on tasks where prediction of a target value is relevant
"""

import pandas as pd
from lens_evals.readout_evals.readout_eval_items import MODEL_CORRECTNESS_CSV

df = pd.read_csv(MODEL_CORRECTNESS_CSV)

model_names = df.columns.tolist()[2:]

averages = df.groupby('eval')[model_names].mean()

print(averages)