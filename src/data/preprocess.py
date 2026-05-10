import pandas as pd

def preprocess_features(df: pd.DataFrame):

    X = df.drop(columns=["EnergyRequestedFromGrid_kW_"])

    y = df["EnergyRequestedFromGrid_kW_"]

    X = pd.get_dummies(
        X,
        columns=["WeatherIcon"],
        drop_first=True
    )

    return X, y