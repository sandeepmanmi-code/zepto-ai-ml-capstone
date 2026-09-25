from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

CSV_PATH = BASE_DIR / "titanic.csv"

sns.set_theme(style="whitegrid")


# ============================================================
# 1. LOAD RAW DATASET EXACTLY ONCE
# ============================================================

print("=" * 80)
print("LOADING TITANIC DATASET")
print("=" * 80)

df = sns.load_dataset("titanic")

# Required offline fallback immediately after loading.
df.to_csv(CSV_PATH, index=False)

print(f"Dataset saved to: {CSV_PATH}")
print(f"Shape: {df.shape}")

print("\nDATA INFO")
print("-" * 80)
df.info()

print("\nDESCRIPTIVE STATISTICS")
print("-" * 80)
print(df.describe(include="all").transpose())


# ============================================================
# 2. MISSING-VALUE PROFILE
# ============================================================

missing = (
    df.isna()
    .mean()
    .mul(100)
    .loc[lambda x: x > 0]
    .sort_values(ascending=False)
)

print("\nMISSING-VALUE PERCENTAGES")
print("-" * 80)

for column, percentage in missing.items():
    print(f"{column}: {percentage:.2f}%")


# ============================================================
# 3. CLEANING
# ============================================================

cleaned = df.copy()

# We explicitly decide strategies from the measured percentages.

missing_rates = cleaned.isna().mean() * 100

for column, rate in missing_rates.items():

    if rate == 0:
        continue

    if rate < 5:
        # Under 5%: drop affected rows.
        cleaned = cleaned.dropna(subset=[column])

    elif rate <= 30:
        # 5%-30%: impute.
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            cleaned[column] = cleaned[column].fillna(
                cleaned[column].median()
            )
        else:
            cleaned[column] = cleaned[column].fillna(
                cleaned[column].mode()[0]
            )

    else:
        # Very high missingness.
        #
        # `deck` is dropped because approximately three quarters
        # of observations are missing, making direct imputation
        # unreliable. It also has a relatively sparse categorical
        # signal compared with the other passenger attributes.
        cleaned = cleaned.drop(columns=[column])


print("\nCLEANED DATASET")
print("-" * 80)
print(f"Shape before cleaning: {df.shape}")
print(f"Shape after cleaning:  {cleaned.shape}")

print("\nRemaining missing values:")
print(cleaned.isna().sum()[lambda x: x > 0])


# ============================================================
# 4. UNIVARIATE ANALYSIS
# ============================================================

def iqr_outlier_count(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = (series < lower) | (series > upper)

    return int(mask.sum()), lower, upper


age_outliers, age_lower, age_upper = iqr_outlier_count(
    cleaned["age"]
)

fare_outliers, fare_lower, fare_upper = iqr_outlier_count(
    cleaned["fare"]
)

print("\nIQR OUTLIER ANALYSIS")
print("-" * 80)

print(
    f"Age: {age_outliers} outliers "
    f"outside [{age_lower:.2f}, {age_upper:.2f}]"
)

print(
    f"Fare: {fare_outliers} outliers "
    f"outside [{fare_lower:.2f}, {fare_upper:.2f}]"
)


# Fare statistics
fare_mean = cleaned["fare"].mean()
fare_median = cleaned["fare"].median()

fare_modes = cleaned["fare"].mode()
fare_mode = fare_modes.iloc[0]

print("\nFARE SUMMARY")
print("-" * 80)
print(f"Mean:   {fare_mean:.4f}")
print(f"Median: {fare_median:.4f}")
print(f"Mode:   {fare_mode:.4f}")

if fare_mean > fare_median > fare_mode:
    fare_skew = "right-skewed"
elif fare_mean < fare_median < fare_mode:
    fare_skew = "left-skewed"
else:
    fare_skew = "approximately symmetric / mixed"

print(f"Distribution: {fare_skew}")


# Histogram + boxplot for age
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(cleaned["age"], kde=True, ax=axes[0])
axes[0].set_title("Age Distribution")

sns.boxplot(x=cleaned["age"], ax=axes[1])
axes[1].set_title("Age Box Plot")

plt.tight_layout()
plt.savefig(ARTIFACT_DIR / "age_hist_box.png", dpi=150)
plt.close()


# Histogram + boxplot for fare
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(cleaned["fare"], kde=True, ax=axes[0])
axes[0].set_title("Fare Distribution")

sns.boxplot(x=cleaned["fare"], ax=axes[1])
axes[1].set_title("Fare Box Plot")

plt.tight_layout()
plt.savefig(ARTIFACT_DIR / "fare_hist_box.png", dpi=150)
plt.close()


# ============================================================
# 5. BIVARIATE ANALYSIS
# ============================================================

print("\nSURVIVAL RATE BY SEX")
print("-" * 80)

survival_by_sex = (
    cleaned.groupby("sex")["survived"]
    .mean()
    .mul(100)
)

print(survival_by_sex)


print("\nSURVIVAL RATE BY PCLASS")
print("-" * 80)

survival_by_pclass = (
    cleaned.groupby("pclass")["survived"]
    .mean()
    .mul(100)
)

print(survival_by_pclass)


print("\nSURVIVAL RATE BY SEX AND PCLASS")
print("-" * 80)

survival_by_sex_class = (
    cleaned.groupby(["sex", "pclass"])["survived"]
    .mean()
    .mul(100)
)

print(survival_by_sex_class)


# Explicit boolean masking demonstration.
female_mask = cleaned["sex"] == "female"
male_mask = cleaned["sex"] == "male"

first_class_mask = cleaned["pclass"] == 1
third_class_mask = cleaned["pclass"] == 3

female_first_class = cleaned[
    female_mask & first_class_mask
]["survived"].mean()

male_third_class = cleaned[
    male_mask & third_class_mask
]["survived"].mean()

print("\nBOOLEAN MASKING EXAMPLES")
print("-" * 80)
print(
    f"Female + first class survival: "
    f"{female_first_class * 100:.2f}%"
)
print(
    f"Male + third class survival: "
    f"{male_third_class * 100:.2f}%"
)


# ============================================================
# 6. SURVIVAL CHARTS
# ============================================================

plt.figure(figsize=(7, 5))
survival_by_sex.plot(kind="bar")
plt.ylabel("Survival Rate (%)")
plt.title("Survival Rate by Sex")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(ARTIFACT_DIR / "survival_by_sex.png", dpi=150)
plt.close()


plt.figure(figsize=(7, 5))
survival_by_pclass.plot(kind="bar")
plt.ylabel("Survival Rate (%)")
plt.title("Survival Rate by Passenger Class")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(ARTIFACT_DIR / "survival_by_class.png", dpi=150)
plt.close()


pivot_survival = (
    cleaned
    .pivot_table(
        values="survived",
        index="pclass",
        columns="sex",
        aggfunc="mean",
    )
    .mul(100)
)

plt.figure(figsize=(8, 5))
sns.heatmap(
    pivot_survival,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu",
)
plt.title("Survival Rate by Sex and Passenger Class")
plt.ylabel("Passenger Class")
plt.xlabel("Sex")
plt.tight_layout()
plt.savefig(
    ARTIFACT_DIR / "survival_sex_class.png",
    dpi=150,
)
plt.close()


# ============================================================
# 7. CORRELATION MATRIX
# ============================================================

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
]

correlation_matrix = cleaned[
    correlation_columns
].corr()

print("\nCORRELATION MATRIX")
print("-" * 80)
print(correlation_matrix)


plt.figure(figsize=(8, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
)

plt.title(
    "Correlation Matrix: Titanic Numeric Analysis Columns"
)

plt.tight_layout()

plt.savefig(
    ARTIFACT_DIR / "correlation_heatmap.png",
    dpi=150,
)

plt.close()


# Find the two strongest absolute off-diagonal correlations.
pairs = []

for i, col1 in enumerate(correlation_columns):
    for j, col2 in enumerate(correlation_columns):
        if i >= j:
            continue

        value = correlation_matrix.loc[col1, col2]

        pairs.append(
            {
                "feature_1": col1,
                "feature_2": col2,
                "correlation": value,
                "absolute_correlation": abs(value),
            }
        )

top_pairs = (
    pd.DataFrame(pairs)
    .sort_values(
        "absolute_correlation",
        ascending=False,
    )
    .head(2)
)

print("\nTWO STRONGEST CORRELATIONS")
print("-" * 80)
print(top_pairs.to_string(index=False))


# ============================================================
# 8. MULTIVARIATE DATA STORY
# ============================================================

# Pair plot
pair_data = cleaned[
    ["survived", "pclass", "age", "fare"]
].dropna()

pair_plot = sns.pairplot(
    pair_data,
    hue="survived",
    diag_kind="hist",
    corner=True,
)

pair_plot.fig.suptitle(
    "Titanic Multivariate Relationships",
    y=1.02,
)

pair_plot.savefig(
    ARTIFACT_DIR / "pairplot.png",
    dpi=150,
)

plt.close("all")


# Age vs fare by survival
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=cleaned,
    x="age",
    y="fare",
    hue="survived",
    style="sex",
    alpha=0.65,
)

plt.title("Age, Fare, Sex and Survival")
plt.tight_layout()

plt.savefig(
    ARTIFACT_DIR / "age_fare_scatter.png",
    dpi=150,
)

plt.close()


# Class / fare / survival boxplot
plt.figure(figsize=(8, 6))

sns.boxplot(
    data=cleaned,
    x="pclass",
    y="fare",
    hue="survived",
)

plt.title("Fare Distribution by Class and Survival")
plt.tight_layout()

plt.savefig(
    ARTIFACT_DIR / "fare_class_survival.png",
    dpi=150,
)

plt.close()


# ============================================================
# 9. EDA-STAGE STANDARDIZATION SANITY CHECK
# ============================================================

eda_scaled = cleaned.copy()

scaler = StandardScaler()

eda_scaled[
    ["age_z", "fare_z"]
] = scaler.fit_transform(
    cleaned[["age", "fare"]]
)

print("\nSTANDARDIZATION CHECK")
print("-" * 80)

print("Before standardization:")
print(
    cleaned[["age", "fare"]]
    .agg(["mean", "std"])
)

print("\nAfter standardization:")
print(
    eda_scaled[["age_z", "fare_z"]]
    .agg(["mean", "std"])
)


# Before / after age
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(
    cleaned["age"],
    kde=True,
    ax=axes[0],
)

axes[0].set_title("Age Before Standardization")

sns.histplot(
    eda_scaled["age_z"],
    kde=True,
    ax=axes[1],
)

axes[1].set_title("Age After Standardization")

plt.tight_layout()

plt.savefig(
    ARTIFACT_DIR / "age_before_after.png",
    dpi=150,
)

plt.close()


# Before / after fare
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(
    cleaned["fare"],
    kde=True,
    ax=axes[0],
)

axes[0].set_title("Fare Before Standardization")

sns.histplot(
    eda_scaled["fare_z"],
    kde=True,
    ax=axes[1],
)

axes[1].set_title("Fare After Standardization")

plt.tight_layout()

plt.savefig(
    ARTIFACT_DIR / "fare_before_after.png",
    dpi=150,
)

plt.close()


# ============================================================
# 10. SAVE EDA RESULTS FOR THE MODELING STAGE
# ============================================================

# The original raw data remains available as titanic.csv.
#
# The modeling stage reads this same committed cleaned dataset.
# We save the cleaned data separately so that the modeling stage
# does not need to make any new raw-data download.

CLEANED_CSV_PATH = BASE_DIR / "titanic_cleaned.csv"

cleaned.to_csv(
    CLEANED_CSV_PATH,
    index=False,
)

print("\nSaved cleaned dataset:")
print(CLEANED_CSV_PATH)


# ============================================================
# 11. GENERATE TEXT REPORT
# ============================================================

report_path = BASE_DIR / "eda_results.md"

with open(report_path, "w", encoding="utf-8") as report:

    report.write("# EDA Results\n\n")

    report.write("## Dataset profile\n\n")
    report.write(f"- Original shape: `{df.shape}`\n")
    report.write(f"- Cleaned shape: `{cleaned.shape}`\n\n")

    report.write("## Missing values\n\n")

    for column, percentage in missing.items():
        if percentage < 5:
            strategy = (
                "Rows containing missing values were dropped because "
                "the missing rate is below 5%."
            )
        elif percentage <= 30:
            strategy = (
                "Missing values were imputed because the missing rate "
                "falls between 5% and 30%."
            )
        else:
            strategy = (
                "The column was dropped because missingness exceeds "
                "30%, making direct imputation unreliable."
            )

        report.write(
            f"- **{column}: {percentage:.2f}%** — {strategy}\n"
        )

    report.write("\n## Outliers\n\n")

    report.write(
        f"- Age: **{age_outliers}** IQR outliers.\n"
    )

    report.write(
        f"- Fare: **{fare_outliers}** IQR outliers.\n"
    )

    report.write("\n## Fare distribution\n\n")

    report.write(
        f"Fare mean = **{fare_mean:.2f}**, "
        f"median = **{fare_median:.2f}**, "
        f"mode = **{fare_mode:.2f}**. "
    )

    if fare_mean > fare_median > fare_mode:
        report.write(
            "The ordering mean > median > mode indicates a "
            "**right-skewed** fare distribution, with higher fares "
            "pulling the mean upward.\n"
        )
    elif fare_mean < fare_median < fare_mode:
        report.write(
            "The ordering mean < median < mode indicates a "
            "**left-skewed** fare distribution.\n"
        )
    else:
        report.write(
            "The mean, median and mode do not follow a strictly "
            "monotonic ordering, so the distribution is described "
            "as approximately symmetric or mixed by this rule.\n"
        )

    report.write("\n## Survival rates\n\n")

    report.write("### By sex\n\n")
    report.write(
        survival_by_sex.to_string()
    )

    report.write("\n\n### By passenger class\n\n")
    report.write(
        survival_by_pclass.to_string()
    )

    report.write("\n\n### By sex and class\n\n")
    report.write(
        survival_by_sex_class.to_string()
    )

    report.write("\n\n## Strongest correlations\n\n")

    for _, row in top_pairs.iterrows():
        direction = (
            "positive"
            if row["correlation"] > 0
            else "negative"
        )

        report.write(
            f"- **{row['feature_1']} and "
            f"{row['feature_2']}**: "
            f"r = {row['correlation']:.3f}. "
            f"This is the stronger pair according to the absolute "
            f"off-diagonal correlation criterion and represents a "
            f"{direction} linear association.\n"
        )

    report.write("\n## Multivariate chart interpretations\n\n")

    report.write(
        "### Survival by sex\n\n"
        "The survival-rate chart shows a clear difference in observed "
        "survival between male and female passengers. This indicates "
        "that sex is an important descriptive variable for explaining "
        "variation in the target. The chart describes an association "
        "and does not by itself establish causation.\n\n"
    )

    report.write(
        "### Survival by passenger class\n\n"
        "Survival rates vary across passenger classes, showing that "
        "passenger class contains substantial information about the "
        "observed survival outcome. The class breakdown also provides "
        "context for the relationship between socioeconomic position "
        "and survival in this dataset.\n\n"
    )

    report.write(
        "### Survival by sex and class\n\n"
        "Combining sex and passenger class reveals interactions that "
        "are hidden when either variable is considered alone. The "
        "survival pattern is therefore more informative when the two "
        "categorical variables are examined jointly rather than "
        "independently.\n\n"
    )

    report.write(
        "### Age and fare scatter plot\n\n"
        "The age-versus-fare scatter plot shows how survival varies "
        "across two continuous passenger attributes while sex is "
        "displayed separately. It provides a multivariate view of "
        "the passenger groups and helps identify whether particular "
        "combinations of age and fare are associated with different "
        "survival outcomes.\n\n"
    )

    report.write(
        "### Fare by class and survival\n\n"
        "The fare distributions differ substantially across passenger "
        "classes, and survival status can be compared within each "
        "class. This chart connects fare, class and survival and "
        "shows why a single-variable fare comparison would miss "
        "important passenger-group structure.\n\n"
    )

    report.write("## Standardization sanity check\n\n")

    report.write(
        "Age and fare were standardized using the z-score formula "
        "`z = (x - mean) / std`. After transformation, both columns "
        "have means approximately equal to zero and standard "
        "deviations approximately equal to one. This transformation "
        "is an EDA sanity check only and is not passed into the "
        "modeling pipeline.\n"
    )

print(f"\nEDA report saved to: {report_path}")
