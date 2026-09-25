from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree


BASE_DIR = Path(__file__).resolve().parent

ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "model_outputs"

ARTIFACT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

CLEANED_DATA = BASE_DIR / "titanic_cleaned.csv"

RANDOM_STATE = 42


# ============================================================
# 1. LOAD THE SAME CLEANED DATASET
# ============================================================

df = pd.read_csv(CLEANED_DATA)

print("=" * 80)
print("MODELING DATA")
print("=" * 80)

print(df.shape)
print(df.head())


# ============================================================
# 2. CLASS BALANCE
# ============================================================

class_counts = df["survived"].value_counts().sort_index()
class_percentages = (
    df["survived"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
)

print("\nCLASS BALANCE")
print("-" * 80)

for class_value in class_counts.index:
    print(
        f"survived={class_value}: "
        f"{class_counts[class_value]} "
        f"({class_percentages[class_value]:.2f}%)"
    )


# ============================================================
# 3. FEATURES AND TARGET
# ============================================================

target = "survived"

# adult_male and alone are deliberately excluded because the
# assignment identifies them as derived/redundant flags.
#
# sex and embarked are categorical features.
# pclass, age, sibsp, parch, fare are numeric features.

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
]

X = df[features]
y = df[target]


# ============================================================
# 4. STRATIFIED TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTRAIN/TEST SPLIT")
print("-" * 80)
print(f"Train size: {len(X_train)}")
print(f"Test size:  {len(X_test)}")

print("\nTraining class distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting class distribution:")
print(y_test.value_counts(normalize=True))


# ============================================================
# 5. PREPROCESSING
# ============================================================

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
]

categorical_features = [
    "sex",
    "embarked",
]


numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ]
)


# ============================================================
# 6. CLASSIFIERS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=RANDOM_STATE,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}


# ============================================================
# 7. TRAIN AND EVALUATE ALL THREE
# ============================================================

classification_results = []
trained_models = {}

confusion_matrices = {}

roc_data = {}

for name, estimator in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )
    auc = roc_auc_score(
        y_test,
        y_probability,
    )

    classification_results.append(
        {
            "model": name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
        }
    )

    trained_models[name] = pipeline

    confusion_matrices[name] = confusion_matrix(
        y_test,
        y_pred,
    )

    fpr, tpr, _ = roc_curve(
        y_test,
        y_probability,
    )

    roc_data[name] = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
    }

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0,
        )
    )

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"AUC:       {auc:.4f}")


classification_df = pd.DataFrame(
    classification_results
)

print("\nCLASSIFICATION COMPARISON")
print("-" * 80)
print(
    classification_df.to_string(
        index=False
    )
)

classification_df.to_csv(
    OUTPUT_DIR / "classification_results.csv",
    index=False,
)


# ============================================================
# 8. CONFUSION MATRICES
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4),
)

for ax, (name, matrix) in zip(
    axes,
    confusion_matrices.items(),
):

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar=False,
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()

plt.savefig(
    ARTIFACT_DIR / "confusion_matrices.png",
    dpi=150,
)

plt.close()


# ============================================================
# 9. ROC CURVES
# ============================================================

plt.figure(figsize=(8, 6))

for name, values in roc_data.items():

    plt.plot(
        values["fpr"],
        values["tpr"],
        label=(
            f"{name} "
            f"(AUC={values['auc']:.3f})"
        ),
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()

plt.tight_layout()

plt.savefig(
    ARTIFACT_DIR / "roc_curves.png",
    dpi=150,
)

plt.close()


# ============================================================
# 10. DECISION TREE VISUALIZATION
# ============================================================

tree_pipeline = trained_models["Decision Tree"]

fitted_preprocessor = tree_pipeline.named_steps[
    "preprocessor"
]

tree_model = tree_pipeline.named_steps[
    "classifier"
]

feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)

plt.figure(figsize=(24, 12))

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True,
    rounded=True,
    fontsize=7,
)

plt.title("Decision Tree Classifier")

plt.tight_layout()

plt.savefig(
    ARTIFACT_DIR / "decision_tree.png",
    dpi=150,
)

plt.close()


# ============================================================
# 11. IMBALANCE COMPARISON
# ============================================================

imbalance_models = {}


# A. Baseline
baseline_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

baseline_pipeline.fit(
    X_train,
    y_train,
)

imbalance_models["Baseline"] = baseline_pipeline


# B. class_weight='balanced'
balanced_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

balanced_pipeline.fit(
    X_train,
    y_train,
)

imbalance_models["Class Weight Balanced"] = balanced_pipeline


# C. SMOTE on training fold only.
smote_pipeline = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "smote",
            SMOTE(
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

smote_pipeline.fit(
    X_train,
    y_train,
)

imbalance_models["SMOTE"] = smote_pipeline


imbalance_results = []

for name, pipeline in imbalance_models.items():

    prediction = pipeline.predict(X_test)

    imbalance_results.append(
        {
            "strategy": name,
            "precision": precision_score(
                y_test,
                prediction,
                zero_division=0,
            ),
            "recall": recall_score(
                y_test,
                prediction,
                zero_division=0,
            ),
            "f1": f1_score(
                y_test,
                prediction,
                zero_division=0,
            ),
        }
    )


imbalance_df = pd.DataFrame(
    imbalance_results
)

print("\nIMBALANCE COMPARISON")
print("-" * 80)
print(imbalance_df.to_string(index=False))

imbalance_df.to_csv(
    OUTPUT_DIR / "imbalance_results.csv",
    index=False,
)


# ============================================================
# 12. RANDOM FOREST GRID SEARCH
# ============================================================

rf_estimator = RandomForestClassifier(
    oob_score=True,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", rf_estimator),
    ]
)

param_grid = {
    "classifier__n_estimators": [
        100,
        200,
        300,
    ],

    "classifier__max_depth": [
        None,
        5,
        10,
    ],

    "classifier__max_features": [
        "sqrt",
        "log2",
        None,
    ],
}

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    refit=True,
)

grid_search.fit(
    X_train,
    y_train,
)

best_rf_pipeline = grid_search.best_estimator_

best_rf = best_rf_pipeline.named_steps[
    "classifier"
]

print("\nRANDOM FOREST GRID SEARCH")
print("-" * 80)
print("Best parameters:")
print(grid_search.best_params_)

print(
    f"Best cross-validation accuracy: "
    f"{grid_search.best_score_:.4f}"
)

print(
    f"OOB score: "
    f"{best_rf.oob_score_:.4f}"
)


# ============================================================
# 13. REGRESSION — PREDICT FARE
# ============================================================

regression_features = [
    "survived",
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "embarked",
]

X_reg = df[regression_features]
y_reg = df["fare"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=RANDOM_STATE,
)


reg_numeric = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
]

reg_categorical = [
    "sex",
    "embarked",
]


reg_numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)


reg_categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)


reg_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            reg_numeric_pipeline,
            reg_numeric,
        ),
        (
            "categorical",
            reg_categorical_pipeline,
            reg_categorical,
        ),
    ]
)


regression_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            reg_preprocessor,
        ),
        (
            "regressor",
            LinearRegression(),
        ),
    ]
)


regression_pipeline.fit(
    Xr_train,
    yr_train,
)

fare_prediction = regression_pipeline.predict(
    Xr_test
)

mae = mean_absolute_error(
    yr_test,
    fare_prediction,
)

rmse = np.sqrt(
    mean_squared_error(
        yr_test,
        fare_prediction,
    )
)

r2 = r2_score(
    yr_test,
    fare_prediction,
)

n = len(yr_test)

# Number of fitted regression coefficients after encoding.
regressor = regression_pipeline.named_steps[
    "regressor"
]

p = len(regressor.coef_)

adjusted_r2 = (
    1
    - (
        (1 - r2)
        * (n - 1)
        / (n - p - 1)
    )
)

regression_results = pd.DataFrame(
    [
        {
            "model": "Multivariate Linear Regression",
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Adjusted_R2": adjusted_r2,
        }
    ]
)

print("\nREGRESSION RESULTS")
print("-" * 80)
print(
    regression_results.to_string(
        index=False
    )
)

regression_results.to_csv(
    OUTPUT_DIR / "regression_results.csv",
    index=False,
)


# ============================================================
# 14. RESIDUAL PLOT
# ============================================================

residuals = yr_test - fare_prediction

plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=fare_prediction,
    y=residuals,
    alpha=0.65,
)

plt.axhline(
    0,
    color="red",
    linestyle="--",
)

plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Linear Regression Residual Plot")

plt.tight_layout()

plt.savefig(
    ARTIFACT_DIR / "regression_residuals.png",
    dpi=150,
)

plt.close()


# ============================================================
# 15. MODEL COMPARISON
# ============================================================

comparison = classification_df.copy()

comparison = comparison.rename(
    columns={
        "accuracy": "Classification Accuracy",
        "precision": "Classification Precision",
        "recall": "Classification Recall",
        "f1": "Classification F1",
        "auc": "Classification AUC",
    }
)

comparison[
    "Regression MAE"
] = np.nan

comparison[
    "Regression RMSE"
] = np.nan

comparison[
    "Regression R2"
] = np.nan

comparison[
    "Regression Adjusted R2"
] = np.nan

# Regression metrics are separate columns and are not
# interpreted on the same scale as classification metrics.

comparison = comparison[
    [
        "model",
        "Classification Accuracy",
        "Classification Precision",
        "Classification Recall",
        "Classification F1",
        "Classification AUC",
        "Regression MAE",
        "Regression RMSE",
        "Regression R2",
        "Regression Adjusted R2",
    ]
]

print("\nMODEL COMPARISON")
print("-" * 80)
print(
    comparison.to_string(
        index=False
    )
)

comparison.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False,
)


# ============================================================
# 16. SELECT AND SAVE COMPLETE PIPELINE
# ============================================================

# Select the model based on F1 as the primary classification
# metric, with AUC as an additional diagnostic.
#
# This selects a COMPLETE preprocessing + estimator pipeline.

best_model_name = (
    classification_df
    .sort_values(
        ["f1", "auc"],
        ascending=False,
    )
    .iloc[0]["model"]
)

best_pipeline = trained_models[
    best_model_name
]

print("\nSELECTED FINAL PIPELINE")
print("-" * 80)
print(f"Model: {best_model_name}")

MODEL_PATH = (
    MODEL_DIR / "best_pipeline.joblib"
)

joblib.dump(
    best_pipeline,
    MODEL_PATH,
)

print(f"Saved to: {MODEL_PATH}")


# ============================================================
# 17. FINAL WRITTEN RECOMMENDATION
# ============================================================

best_row = (
    classification_df[
        classification_df["model"]
        == best_model_name
    ]
    .iloc[0]
)

recommendation_path = (
    BASE_DIR / "model_recommendation.md"
)

with open(
    recommendation_path,
    "w",
    encoding="utf-8",
) as file:

    file.write("# Model Recommendation\n\n")

    file.write(
        f"Based on the held-out test-set comparison, "
        f"**{best_model_name}** was selected as the final "
        f"classification pipeline using F1 as the primary "
        f"selection metric and AUC as an additional diagnostic. "
        f"Its test accuracy was "
        f"**{best_row['accuracy']:.3f}**, precision was "
        f"**{best_row['precision']:.3f}**, recall was "
        f"**{best_row['recall']:.3f}**, F1 was "
        f"**{best_row['f1']:.3f}**, and AUC was "
        f"**{best_row['auc']:.3f}**. "
    )

    file.write(
        "F1 was used as the primary selection criterion because "
        "it balances precision and recall, while AUC provides an "
        "additional threshold-independent view of ranking "
        "performance. The final artifact contains the complete "
        "preprocessing and estimator pipeline, so new raw "
        "records can be passed directly to the saved object. "
        "The regression metrics are reported separately because "
        "MAE, RMSE, R² and adjusted R² measure a continuous "
        "prediction problem and are not directly comparable with "
        "classification accuracy, precision, recall, F1 or AUC.\n"
    )

print(
    f"\nRecommendation saved to: "
    f"{recommendation_path}"
)
