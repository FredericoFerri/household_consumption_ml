import numpy as np
import pandas as pd

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:

    # Corrupted rows
    bad_rows = [7677, 22467, 35949]
    df = df.drop(index=bad_rows)

    # Remove Inf and missing values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df