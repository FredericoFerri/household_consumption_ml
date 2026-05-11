import pandas as pd
from src.data.load_data import load_raw_data
from src.data.cleaning import clean_dataset
from src.data.preprocess import preprocess_features

from src.models.train_linear_regression import train_linear_regression
from src.models.train_random_forest import train_random_forest
from src.models.evaluate import evaluate_model, evaluate_both_regimes, plot_residuals, plot_predicted_vs_actual

from sklearn.model_selection import cross_val_score


def main():

    df = load_raw_data("data/raw/household_energy_data.csv")
    print("Dataset loaded.")

    df = clean_dataset(df)
    print("Dataset cleaned.")

    X, y = preprocess_features(df)
    print("Features preprocessed.")

    # --- Linear Regression ---
    print("\n========== Linear Regression ==========")
    model_lr, scaler, y_pred_lr, y_test_lr, test_indices_lr = train_linear_regression(X, y)
    print("Model trained.")
    evaluate_model(y_test_lr, y_pred_lr, label="Dataset")
    evaluate_both_regimes(y_test_lr, y_pred_lr, test_indices_lr, df)

    print("\nCoefficients:")
    coefficients = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model_lr.coef_
    }).sort_values(by="Coefficient", ascending=False)
    print(coefficients)

    print("\nCross-validation results:")
    scores_lr = cross_val_score(model_lr, X, y, cv=5, scoring="r2")
    print(scores_lr)
    print("Mean R_2:", scores_lr.mean())

    # Plottings for 
    #plot_predicted_vs_actual(y_test_lr, y_pred_lr, label="Linear Regression")
    #plot_residuals(y_test_lr, y_pred_lr, label="Linear Regression")

    # --- Linear Regression: only central heating (leakage check) ---
    print("\n========== Linear Regression — Central Heating only ==========")
    X_leaking = df[["CentralHeating_kW_"]]
    y_leaking = df["EnergyRequestedFromGrid_kW_"]
    _, scaler_leaking, y_pred_leaking, y_test_leaking, test_indices_leaking = train_linear_regression(X_leaking, y_leaking)
    evaluate_model(y_test_leaking, y_pred_leaking, label="Central Heating Only")

    # --- Linear Regression: no central heating (heating-driven check) ---
    print("\n========== Linear Regression — No Central Heating ==========")
    X_no_heating = X.drop(columns=["CentralHeating_kW_"])
    _, scaler_nh, y_pred_nh, y_test_nh, test_indices_nh = train_linear_regression(X_no_heating, y)
    print("Model trained.")
    evaluate_model(y_test_nh, y_pred_nh, label="No Central Heating")
    evaluate_both_regimes(y_test_nh, y_pred_nh, test_indices_nh, df)

    # --- Random Forest ---
    print("\n========== Random Forest ==========")
    model_rf, y_pred_rf, y_test_rf, test_indices_rf = train_random_forest(X, y)
    print("Model trained.")
    evaluate_model(y_test_rf, y_pred_rf, label="Dataset")
    evaluate_both_regimes(y_test_rf, y_pred_rf, test_indices_rf, df)

    # --- Random Forest: no central heating ---
    print("\n========== Random Forest — No Central Heating ==========")
    X_no_heating_rf = X.drop(columns=["CentralHeating_kW_"])
    model_rf_nh, y_pred_rf_nh, y_test_rf_nh, test_indices_rf_nh = train_random_forest(X_no_heating_rf, y)
    print("Model trained.")
    evaluate_model(y_test_rf_nh, y_pred_rf_nh, label="Dataset - No central heating")
    evaluate_both_regimes(y_test_rf_nh, y_pred_rf_nh, test_indices_rf_nh, df)


if __name__ == "__main__":
    main()