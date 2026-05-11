from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

APPLIANCE_COLS = [
    'VentilationAndAC_kW_',
    'CentralHeating_kW_',
    'Dishwasher_kW_',
    'UnderFloorHeater_kW_',
    'ElectricRadiator_kW_',
    'HomeOffice_kW_',
    'Fridge_kW_',
    'CellarSumpPump_kW_',
    'GarageDoor_kW_',
    'Oven_kW_',
    'CeramicHob_kW_',
    'Kettle_kW_',
    'MechanicalLoftVentilator_kW_',
    'ShowerPump_kW_',
    'Microwave_kW_',
    'LivingRoom_kW_'
]

def plot_linear_regression(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))

    sns.regplot(
        x=df["net_consumption"],
        y=df["EnergyRequestedFromGrid_kW_"],
        scatter_kws={"alpha": 0.4},
        line_kws={"linewidth": 2, "color": "red"}
    )

    plt.xlabel("Net Consumption (kW)")
    plt.ylabel("Energy Requested From Grid (kW)")

    plt.title(
        "Grid Demand vs Net Consumption"
    )

    plt.show()

def plot_predicted_vs_actual(y_test, y_pred, label="Linear Regression"):
    y_test_vals = y_test.values if hasattr(y_test, "values") else y_test

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_vals, y_pred, alpha=0.3, s=10, color="steelblue", label="Predictions")

    min_val = min(y_test_vals.min(), y_pred.min())
    max_val = max(y_test_vals.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linewidth=2, label="Perfect fit")

    plt.xlabel("Actual Energy Requested From Grid (kW)")
    plt.ylabel("Predicted Energy Requested From Grid (kW)")
    plt.title(f"Predicted vs Actual — {label}")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_residuals(y_test, y_pred, label="Linear Regression"):
    y_test_vals = y_test.values if hasattr(y_test, "values") else y_test
    residuals = y_test_vals - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10, color="steelblue")
    axes[0].axhline(0, color="red", linewidth=2)
    axes[0].set_xlabel("Predicted (kW)")
    axes[0].set_ylabel("Residual (kW)")
    axes[0].set_title(f"Residuals vs Predicted — {label}")

    # Residuals distribution
    axes[1].hist(residuals, bins=60, color="steelblue", edgecolor="white")
    axes[1].axvline(0, color="red", linewidth=2)
    axes[1].set_xlabel("Residual (kW)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Residuals Distribution — {label}")

    plt.tight_layout()
    plt.show()

def evaluate_model(y_test, y_pred, label="Dataset"):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- {label} ---")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R_2:", r2)

def evaluate_both_regimes(y_test, y_pred, test_indices, df: pd.DataFrame):

    df_plot = df.loc[test_indices].copy()

    df_plot["y_test"] = y_test
    df_plot["y_pred"] = y_pred

    df_plot["appliance_sum"] = df_plot[APPLIANCE_COLS].sum(axis=1)

    df_plot["net_consumption"] = (df_plot["appliance_sum"] - df_plot["Solar_kW_"])

    low_consumption = df_plot[df_plot["net_consumption"] <= 7] 
    high_consumption = df_plot[df_plot["net_consumption"] > 7]

    evaluate_model( 
        low_consumption["y_test"], 
        low_consumption["y_pred"], 
        label="Net Consumption <= 7" ) 
    
    evaluate_model( 
        high_consumption["y_test"], 
        high_consumption["y_pred"], 
        label="Net Consumption > 7" )