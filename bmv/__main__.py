"""python -m bmv [out_dir]: run every sweep and write results/exp*.csv."""

import sys
from pathlib import Path

import pandas as pd

from .experiments import ALL

out = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
out.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 250)
for i, exp in enumerate(ALL, 1):
    df = exp()
    df.to_csv(out / f"exp{i}_{df['experiment'][0]}.csv", index=False)
    print(df.drop(columns="experiment").round(4).to_string(index=False), "\n", flush=True)
