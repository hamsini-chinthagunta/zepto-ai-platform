
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from imblearn.over_sampling import SMOTE

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------
# Folders and dataset loading
# ---------------------------------------------------------

BASE = Path(__file__).resolve().parent
PLOTS = BASE / "plots"
MODELS = BASE / "models"

PLOTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

# Load Titanic dataset exactly once and immediately save fallback.
df = sns.load_dataset("titanic")
df.to_csv(BASE / "titanic.csv", index=False)

print("=" * 60)
print("TITANIC DATASET PROFILE")
print("=" * 60)

print("\nShape:", df.shape)

print("\nDataset info:")
df.info()

print("\nDescriptive statistics:")
print(df.describe(include="all").to_string())

missing_pct = df.isna().mean() * 100
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)

print("\nMissing-value percentages:")
print(missing_pct.round(2).to_string())


# ---------------------------------------------------------
# Missing-value treatment
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MISSING-VALUE TREATMENT")
print("=" * 60)

clean = df.copy()

# deck has very high missingness, so remove the column.
if "deck" in clean.columns:
    print(
        f"deck: {clean['deck'].isna().mean() * 100:.2f}% missing. "
        "Dropping this column because most entries are missing."
    )
    clean = clean.drop(columns=["deck"])

# age has moderate missingness, so fill with the median.
age_missing = clean["age"].isna().sum()
age_median = clean["age"].median()
clean["age"] = clean["age"].fillna(age_median)

print(
    f"age: {age_missing} missing values. "
    f"Imputed with median age {age_median:.2f}."
)

# embarked and embark_town have less than 5% missingness.
# Drop rows with missing values in these columns.
for col in ["embarked", "embark_town"]:
    if col in clean.columns:
        before = len(clean)
        clean = clean.dropna(subset=[col])
        removed = before - len(clean)
        print(
            f"{col}: under 5% missing. "
            f"Dropped {removed} row(s) with missing {col}."
        )

# Remove any rows missing the target or core numerical fields.
before = len(clean)
clean = clean.dropna(subset=["survived", "pclass", "fare"])
print(f"Dropped {before - len(clean)} additional row(s) missing core fields.")
print("Rows remaining after cleaning:", len(clean))


# ---------------------------------------------------------
# EDA: distributions and IQR outliers
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DISTRIBUTIONS AND OUTLIERS")
print("=" * 60)

for col in ["age", "fare"]:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    sns.histplot(data=clean, x=col, kde=True, ax=axes[0])
    axes[0].set_title(f"{col.title()} distribution")

    sns.boxplot(data=clean, x=col, ax=axes[1])
    axes[1].set_title(f"{col.title()} boxplot")

    fig.tight_layout()
    fig.savefig(PLOTS / f"{col}_hist_box.png", dpi=150)
    plt.close(fig)

    q1 = clean[col].quantile(0.25)
    q3 = clean[col].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_count = (
        (clean[col] < lower_bound) | (clean[col] > upper_bound)
    ).sum()

    print(
        f"{col}: Q1={q1:.2f}, Q3={q3:.2f}, "
        f"IQR={iqr:.2f}, outlier count={outlier_count}"
    )


# ---------------------------------------------------------
# Fare statistics
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FARE STATISTICS")
print("=" * 60)

fare = clean["fare"]

fare_mean = fare.mean()
fare_median = fare.median()
fare_modes = fare.mode().tolist()
fare_skewness = fare.skew()

print(f"Fare mean: {fare_mean:.4f}")
print(f"Fare median: {fare_median:.4f}")
print(f"Fare mode(s): {fare_modes}")
print(f"Fare skewness: {fare_skewness:.4f}")

if fare_skewness > 0:
    print(
        "Interpretation: Fare is positively (right) skewed. "
        "A smaller number of high fares pull the mean upward, "
        "so the median better represents a typical passenger fare."
    )
elif fare_skewness < 0:
    print(
        "Interpretation: Fare is negatively (left) skewed. "
        "The longer tail is toward lower fares, which can pull "
        "the mean below the median."
    )
else:
    print(
        "Interpretation: Fare skewness is close to zero, "
        "suggesting a roughly symmetric distribution."
    )


# ---------------------------------------------------------
# Survival-rate breakdowns using boolean masks
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("SURVIVAL-RATE BREAKDOWNS")
print("=" * 60)

print("\nSurvival rate by sex:")
for sex in sorted(clean["sex"].dropna().unique()):
    mask = clean["sex"] == sex
    rate = clean.loc[mask, "survived"].mean()
    print(f"{sex}: {rate:.4f}")

print("\nSurvival rate by passenger class:")
for passenger_class in sorted(clean["pclass"].dropna().unique()):
    mask = clean["pclass"] == passenger_class
    rate = clean.loc[mask, "survived"].mean()
    print(f"Class {passenger_class}: {rate:.4f}")

print("\nSurvival rate by sex and passenger class:")
for sex in sorted(clean["sex"].dropna().unique()):
    for passenger_class in sorted(clean["pclass"].dropna().unique()):
        mask = (
            (clean["sex"] == sex)
            & (clean["pclass"] == passenger_class)
        )

        if mask.sum() > 0:
            rate = clean.loc[mask, "survived"].mean()
            print(
                f"{sex}, class {passenger_class}: {rate:.4f} "
                f"(n={mask.sum()})"
            )


# ---------------------------------------------------------
# Correlation: exactly the six requested numeric columns
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = clean[corr_cols].corr(numeric_only=True)

print("\nCorrelation matrix:")
print(corr.round(3).to_string())

# Find the strongest absolute correlations, excluding the diagonal.
upper_triangle = np.triu(np.ones(corr.shape), k=1).astype(bool)
absolute_correlations = corr.abs().where(upper_triangle)

strongest_pairs = (
    absolute_correlations.stack()
    .sort_values(ascending=False)
    .head(2)
)

print("\nTwo strongest absolute correlations:")
print(strongest_pairs.to_string())


# ---------------------------------------------------------
# Charts with written interpretations
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CHARTS AND INTERPRETATIONS")
print("=" * 60)

# Chart 1: survival by sex
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=clean, x="sex", y="survived", ax=ax)
ax.set_title("Observed survival rate by sex")
ax.set_xlabel("Sex")
ax.set_ylabel("Survival rate")
fig.tight_layout()
fig.savefig(PLOTS / "survival_by_sex.png", dpi=150)
plt.close(fig)

print(
    "\nChart 1 interpretation: The chart compares the observed survival "
    "rates of female and male passengers in this dataset. The difference "
    "is descriptive and does not, by itself, explain why any individual "
    "passenger survived."
)

# Chart 2: survival by passenger class
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=clean, x="pclass", y="survived", ax=ax)
ax.set_title("Observed survival rate by passenger class")
ax.set_xlabel("Passenger class")
ax.set_ylabel("Survival rate")
fig.tight_layout()
fig.savefig(PLOTS / "survival_by_class.png", dpi=150)
plt.close(fig)

print(
    "\nChart 2 interpretation: The chart displays the observed survival "
    "rate for each passenger class. It shows how survival rates differed "
    "across these groups in the available Titanic records, without "
    "establishing that class alone caused the difference."
)

# Chart 3: age distribution by survival
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(
    data=clean,
    x="age",
    hue="survived",
    bins=25,
    element="step",
    stat="count",
    common_norm=False,
    ax=ax
)
ax.set_title("Age distribution by survival")
ax.set_xlabel("Age")
ax.set_ylabel("Passenger count")
fig.tight_layout()
fig.savefig(PLOTS / "age_by_survival.png", dpi=150)
plt.close(fig)

print(
    "\nChart 3 interpretation: The overlapping histograms compare the "
    "age distributions of passengers who survived and those who did not. "
    "The distributions show which age ranges appear in each group, "
    "but age alone should not be treated as an explanation of survival."
)

# Chart 4: fare distribution by survival
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=clean, x="survived", y="fare", ax=ax)
ax.set_title("Fare distribution by survival")
ax.set_xlabel("Survived (0 = no, 1 = yes)")
ax.set_ylabel("Fare")
fig.tight_layout()
fig.savefig(PLOTS / "fare_by_survival.png", dpi=150)
plt.close(fig)

print(
    "\nChart 4 interpretation: The boxplots compare the median, spread, "
    "and potential outliers in fares for the two survival groups. "
    "Any difference is descriptive and may be associated with other "
    "passenger characteristics, such as passenger class."
)


# ---------------------------------------------------------
# Classification: preprocessing and stratified split
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION MODELS")
print("=" * 60)

target = clean["survived"].astype(int)
features = clean.drop(columns=["survived", "alive"], errors="ignore")

numeric_features = features.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = features.select_dtypes(
    exclude=["number"]
).columns.tolist()

numeric_prep = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_prep = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_prep, numeric_features),
    ("cat", categorical_prep, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42,
    stratify=target
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).round(3).to_string())


# ---------------------------------------------------------
# Train and compare baseline classification models
# ---------------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        oob_score=True,
        bootstrap=True
    )
}

model_results = []

for model_name, model in models.items():
    pipeline = Pipeline([
        ("preprocess", clone(preprocessor)),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    model_results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        )
    })

    if model_name == "Random Forest":
        print(
            "\nBaseline Random Forest OOB score:",
            pipeline.named_steps["model"].oob_score_
        )

comparison = pd.DataFrame(model_results).sort_values(
    by="F1", ascending=False
)

print("\nBaseline model comparison:")
print(comparison.to_string(index=False))


# ---------------------------------------------------------
# Class imbalance: original vs class weights vs SMOTE
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CLASS IMBALANCE EXPERIMENT")
print("=" * 60)

imbalance_results = []

# Baseline Random Forest
original_rf = Pipeline([
    ("preprocess", clone(preprocessor)),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        oob_score=True,
        bootstrap=True
    ))
])

original_rf.fit(X_train, y_train)
original_pred = original_rf.predict(X_test)

imbalance_results.append({
    "Method": "Original training data",
    "Accuracy": accuracy_score(y_test, original_pred),
    "Precision": precision_score(
        y_test, original_pred, zero_division=0
    ),
    "Recall": recall_score(
        y_test, original_pred, zero_division=0
    ),
    "F1": f1_score(
        y_test, original_pred, zero_division=0
    )
})

print("\nOriginal Random Forest report:")
print(classification_report(
    y_test, original_pred, zero_division=0
))

# Class-weighted Random Forest
weighted_rf = Pipeline([
    ("preprocess", clone(preprocessor)),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        oob_score=True,
        bootstrap=True
    ))
])

weighted_rf.fit(X_train, y_train)
weighted_pred = weighted_rf.predict(X_test)

imbalance_results.append({
    "Method": "Class-weight balanced",
    "Accuracy": accuracy_score(y_test, weighted_pred),
    "Precision": precision_score(
        y_test, weighted_pred, zero_division=0
    ),
    "Recall": recall_score(
        y_test, weighted_pred, zero_division=0
    ),
    "F1": f1_score(
        y_test, weighted_pred, zero_division=0
    )
})

print("\nClass-weight balanced Random Forest report:")
print(classification_report(
    y_test, weighted_pred, zero_division=0
))

# SMOTE: fit preprocessing on training data only.
# Apply resampling only to the transformed training set.
smote_preprocessor = clone(preprocessor)

X_train_prepared = smote_preprocessor.fit_transform(X_train)
X_test_prepared = smote_preprocessor.transform(X_test)

print("\nTraining class counts before SMOTE:")
print(y_train.value_counts().to_string())

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(
    X_train_prepared,
    y_train
)

print("\nTraining class counts after SMOTE:")
print(pd.Series(y_train_smote).value_counts().to_string())

smote_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    oob_score=True,
    bootstrap=True
)

smote_rf.fit(X_train_smote, y_train_smote)
smote_pred = smote_rf.predict(X_test_prepared)

imbalance_results.append({
    "Method": "SMOTE on training data",
    "Accuracy": accuracy_score(y_test, smote_pred),
    "Precision": precision_score(
        y_test, smote_pred, zero_division=0
    ),
    "Recall": recall_score(
        y_test, smote_pred, zero_division=0
    ),
    "F1": f1_score(
        y_test, smote_pred, zero_division=0
    )
})

print("\nSMOTE Random Forest report:")
print(classification_report(
    y_test, smote_pred, zero_division=0
))

imbalance_comparison = pd.DataFrame(
    imbalance_results
).sort_values(by="F1", ascending=False)

print("\nImbalance-method comparison:")
print(imbalance_comparison.to_string(index=False))


# ---------------------------------------------------------
# Random Forest GridSearchCV
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 60)

tune_pipeline = Pipeline([
    ("preprocess", clone(preprocessor)),
    ("model", RandomForestClassifier(
        random_state=42,
        oob_score=True,
        bootstrap=True
    ))
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__max_features": ["sqrt", "log2"]
}

grid = GridSearchCV(
    estimator=tune_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    refit=True
)

grid.fit(X_train, y_train)

print("\nBest parameters:")
print(grid.best_params_)

print("\nBest cross-validation F1:", grid.best_score_)

best_pipeline = grid.best_estimator_
best_predictions = best_pipeline.predict(X_test)

print("\nTuned Random Forest test report:")
print(classification_report(
    y_test, best_predictions, zero_division=0
))

print(
    "Tuned Random Forest OOB score:",
    best_pipeline.named_steps["model"].oob_score_
)

model_path = MODELS / "best_pipeline.joblib"
joblib.dump(best_pipeline, model_path)

print(f"\nSaved fitted classification pipeline to: {model_path}")


# ---------------------------------------------------------
# Regression side-task: predict fare
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FARE REGRESSION")
print("=" * 60)

reg_features = clean.drop(
    columns=["fare", "alive"],
    errors="ignore"
)
reg_target = clean["fare"]

reg_numeric_features = reg_features.select_dtypes(
    include=["number"]
).columns.tolist()

reg_categorical_features = reg_features.select_dtypes(
    exclude=["number"]
).columns.tolist()

reg_preprocessor = ColumnTransformer([
    (
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        reg_numeric_features
    ),
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]),
        reg_categorical_features
    )
])

regression_pipeline = Pipeline([
    ("preprocess", reg_preprocessor),
    ("model", LinearRegression())
])

RX_train, RX_test, ry_train, ry_test = train_test_split(
    reg_features,
    reg_target,
    test_size=0.2,
    random_state=42
)

regression_pipeline.fit(RX_train, ry_train)
fare_predictions = regression_pipeline.predict(RX_test)

mae = mean_absolute_error(ry_test, fare_predictions)
rmse = np.sqrt(mean_squared_error(ry_test, fare_predictions))
r2 = r2_score(ry_test, fare_predictions)

# Count features after preprocessing, including one-hot columns.
p = regression_pipeline.named_steps["preprocess"].transform(
    RX_test
).shape[1]
n = len(ry_test)

if n > p + 1:
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
else:
    adjusted_r2 = np.nan

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R-squared: {r2:.4f}")
print(f"Adjusted R-squared: {adjusted_r2:.4f}")


# ---------------------------------------------------------
# Residual plot and heteroscedasticity discussion
# ---------------------------------------------------------

residuals = ry_test - fare_predictions

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(fare_predictions, residuals, alpha=0.7)
ax.axhline(y=0, linestyle="--")
ax.set_xlabel("Predicted fare")
ax.set_ylabel("Residual (actual - predicted)")
ax.set_title("Fare regression residual plot")
fig.tight_layout()
fig.savefig(PLOTS / "fare_regression_residuals.png", dpi=150)
plt.close(fig)

print(
    "\nResidual plot interpretation: Examine whether the residuals "
    "form a roughly even band around zero or whether their spread "
    "changes as predicted fare increases. A funnel-shaped or widening "
    "pattern can indicate heteroscedasticity, while a fairly even "
    "spread provides less visual evidence of it."
)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print("Raw fallback CSV:", BASE / "titanic.csv")
print("Charts saved in:", PLOTS)
print("Fitted classification pipeline:", model_path)