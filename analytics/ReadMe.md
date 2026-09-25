Analytics Pipeline — Zepto
Overview

This module implements an end-to-end analytics and machine-learning workflow using the classic Titanic dataset.

The workflow is deliberately connected:

load raw dataset once
        ↓
save offline fallback
        ↓
profile
        ↓
clean
        ↓
EDA
        ↓
train/test split
        ↓
train-only preprocessing
        ↓
classification
        ↓
imbalance analysis
        ↓
hyperparameter tuning
        ↓
regression side-task
        ↓
save complete model pipeline


The raw Titanic dataset is loaded from Seaborn exactly once in 01_eda.py:

df = sns.load_dataset("titanic")
df.to_csv("titanic.csv", index=False)


02_modeling.py does not call sns.load_dataset() again. It continues from the saved CSV produced by the first stage.

Installation

From the repository root:

pip install -r analytics/requirements.txt

Running the module

Run the EDA stage first:

python analytics/01_eda.py


This creates:

analytics/titanic.csv

analytics/titanic_cleaned.csv

EDA charts under analytics/artifacts/

analytics/eda_results.md

Then run the modeling stage:

python analytics/02_modeling.py


This creates:

classification metrics

imbalance comparison

Random Forest grid-search results

regression metrics

residual plot

model comparison

final model recommendation

analytics/models/best_pipeline.joblib

Finally, verify that the saved pipeline works on raw input:

python analytics/reload_pipeline.py

Dataset loading and offline fallback

The dataset is loaded from Seaborn once. Immediately after loading, the unmodified dataframe is written to:

analytics/titanic.csv


This file is committed to the repository and provides the required offline fallback.

The modeling stage uses the cleaned CSV produced by the EDA stage rather than downloading the Titanic dataset again.

Missing-value handling

Missingness is measured immediately after loading and reported as a percentage for every affected column.

The following threshold is used:

Less than 5% missing: rows containing the missing value are dropped.

5%–30% missing: the missing values are imputed.

More than 30% missing: the column is considered too incomplete for reliable direct imputation and is dropped.

For numeric columns in the 5%–30% range, the median is used. For categorical columns, the most frequent category is used.

The exact percentages measured by the submitted dataset are generated in eda_results.md, so the reported values are directly reproducible from the source data.

The high-missingness deck column is dropped rather than filled with an invented value because its missingness is too extensive for ordinary imputation to be reliable.

Univariate analysis

Histograms and box plots are produced for both age and fare.

The IQR rule is:

lower bound = Q1 - 1.5 × IQR
upper bound = Q3 + 1.5 × IQR


Values outside those bounds are counted as outliers.

The exact outlier counts are calculated and written to eda_results.md.

For fare, mean, median and mode are also calculated. The distribution is interpreted using their ordering: when mean > median > mode, the distribution is described as right-skewed.

Bivariate analysis

Survival rates are calculated for:

sex

passenger class

sex and passenger class jointly

Boolean masking is explicitly demonstrated with expressions such as:

cleaned[
    (cleaned["sex"] == "female")
    & (cleaned["pclass"] == 1)
]


This provides a reproducible demonstration of combined boolean conditions.

Correlation analysis

The correlation matrix contains exactly these six columns:

survived
pclass
age
sibsp
parch
fare


adult_male and alone are intentionally excluded because they are derived/redundant flags rather than independent measured features.

The two strongest correlations are identified by ranking all off-diagonal pairs by absolute correlation coefficient.

The exact pairs and coefficients are written to eda_results.md.

Multivariate data story

The analysis includes several complementary charts.

Survival by sex

The survival-rate chart shows differences in observed survival between male and female passengers. This indicates that sex contains substantial information about the survival outcome in this dataset. The chart describes an association rather than establishing a causal relationship.

Survival by passenger class

Survival rates differ across passenger classes. This demonstrates that passenger class contains useful information for understanding the survival outcome and provides socioeconomic context for the observed passenger groups.

Survival by sex and class

Combining sex and passenger class reveals patterns that are not visible from either variable independently. Examining both variables jointly therefore gives a more detailed description of the passenger groups associated with different survival rates.

Age and fare scatter plot

The scatter plot combines age and fare while using survival and sex as additional visual dimensions. This makes it possible to examine several passenger characteristics simultaneously rather than relying on isolated univariate relationships.

Fare by class and survival

Fare distributions vary substantially between passenger classes. Comparing survival within each class shows why fare alone should not be interpreted independently of passenger class.

Correlation heatmap

The heatmap summarizes linear associations among the six specified numeric variables. The two largest absolute off-diagonal coefficients are reported separately so that the strongest observed relationships can be interpreted without treating correlation as causation.

EDA standardization check

Age and fare are standardized during EDA using:

z = (x - mean) / standard deviation


The resulting transformed variables should have approximately:

mean = 0
standard deviation = 1


The before/after summaries and plots confirm this.

This standardization is not used as the modeling pipeline's preprocessing. The predictive pipeline independently fits its own StandardScaler on the training split only.

Classification

The target is:

survived


The features are:

pclass
sex
age
sibsp
parch
fare
embarked


The adult_male and alone flags are excluded because they are derived/redundant variables.

A stratified 80/20 train/test split is used.

Stratification is important because the survival target contains two classes with different frequencies. It keeps approximately the same survived/not-survived proportions in both training and test sets, making the held-out evaluation more representative of the original cleaned dataset.

Preprocessing

Numeric features are processed using:

median imputation → StandardScaler


Categorical features are processed using:

most-frequent imputation → OneHotEncoder


All preprocessing is encapsulated inside a scikit-learn Pipeline/ColumnTransformer.

The preprocessing objects are fitted only on the training data. The test data is transformed using those fitted objects and is never used to fit an imputer, encoder or scaler.

Classification models

Three classifiers are trained on the identical train/test split:

Logistic Regression

Decision Tree

Random Forest

Each model is evaluated using:

confusion matrix

accuracy

precision

recall

F1

ROC curve

AUC

The numerical results are stored in:

analytics/model_outputs/classification_results.csv


The confusion matrices and ROC curves are saved as supporting chart artifacts.

Decision tree

The fitted Decision Tree is visualized using plot_tree.

The visualization includes:

transformed feature names

class names

tree structure

node conditions

The generated chart is:

analytics/artifacts/decision_tree.png

Imbalance comparison

The target class balance is reported before modeling.

A Logistic Regression model is evaluated using three approaches:

baseline with no imbalance handling

class_weight="balanced"

SMOTE applied only to the training data

SMOTE is implemented inside an imblearn pipeline. Therefore synthetic samples are created only when the training pipeline is fitted; the test set is never oversampled.

Precision, recall and F1 are compared in:

analytics/model_outputs/imbalance_results.csv


The appropriate strategy is assessed from the observed precision/recall/F1 trade-off rather than assuming that one imbalance technique must always be superior.

Random Forest tuning

GridSearchCV searches over:

n_estimators
max_depth
max_features


The Random Forest is explicitly constructed with:

RandomForestClassifier(
    oob_score=True,
    ...
)


This ensures that the fitted estimator exposes oob_score_.

The best parameter combination, cross-validation score and OOB score are printed by the modeling script.

The OOB score provides an additional internal estimate of generalization based on observations not included in each bootstrap sample.

Regression side-task

A multivariate linear regression predicts fare using the other available passenger features.

The regression model uses its own preprocessing pipeline with numeric imputation/scaling and categorical imputation/one-hot encoding.

The following metrics are reported:

MAE

RMSE

R²

Adjusted R²

The adjusted R² is calculated as:

Adjusted R² =
1 - ((1 - R²) × (n - 1) / (n - p - 1))


where n is the number of test observations and p is the number of fitted regression coefficients after categorical encoding.

The residual plot is used to assess heteroscedasticity. A roughly random, similarly sized residual spread is consistent with approximately constant variance, whereas a funnel or systematic spread indicates potential heteroscedasticity.

Model comparison

Classification and regression metrics are kept as separate groups.

Classification metrics:

Accuracy
Precision
Recall
F1
AUC


Regression metrics:

MAE
RMSE
R²
Adjusted R²


They are not treated as one common numerical scale because classification and continuous regression evaluate different prediction problems.

The complete comparison is saved to:

analytics/model_outputs/model_comparison.csv

Final classifier selection

The final classification pipeline is selected using F1 as the primary criterion and AUC as an additional diagnostic.

The actual values produced by the run are written to:

analytics/model_recommendation.md


The recommendation considers the observed accuracy, precision, recall, F1 and AUC rather than selecting a model solely from one metric.

Saved production pipeline

The final fitted object is saved as:

analytics/models/best_pipeline.joblib


The saved object is the complete pipeline, not merely the classifier.

It contains:

raw input
   ↓
imputation
   ↓
encoding
   ↓
scaling
   ↓
trained classifier


Therefore new raw records can be supplied directly to:

model.predict(raw_dataframe)


without manually repeating preprocessing.

reload_pipeline.py loads the artifact using joblib.load() and demonstrates prediction on an unprocessed passenger record.