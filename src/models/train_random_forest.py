import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold


def train_random_forest(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, X.index, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- Cross-validation ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=-1)
    print("\nCross-validation results:")
    print(cv_scores)
    print(f"Mean R_2: {cv_scores.mean()}")

    # --- Feature importances ---
    importances = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    print("\n--- Feature Importances ---")
    print(importances.to_string())

    return model, y_pred, y_test, test_indices