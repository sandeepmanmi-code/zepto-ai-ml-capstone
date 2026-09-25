# EDA Results

## Dataset profile

- Original shape: `(891, 15)`
- Cleaned shape: `(889, 14)`

## Missing values

- **deck: 77.22%** — The column was dropped because missingness exceeds 30%, making direct imputation unreliable.
- **age: 19.87%** — Missing values were imputed because the missing rate falls between 5% and 30%.
- **embarked: 0.22%** — Rows containing missing values were dropped because the missing rate is below 5%.
- **embark_town: 0.22%** — Rows containing missing values were dropped because the missing rate is below 5%.

## Outliers

- Age: **65** IQR outliers.
- Fare: **114** IQR outliers.

## Fare distribution

Fare mean = **32.10**, median = **14.45**, mode = **8.05**. The ordering mean > median > mode indicates a **right-skewed** fare distribution, with higher fares pulling the mean upward.

## Survival rates

### By sex

sex
female    74.038462
male      18.890815

### By passenger class

pclass
1    62.616822
2    47.282609
3    24.236253

### By sex and class

sex     pclass
female  1         96.739130
        2         92.105263
        3         50.000000
male    1         36.885246
        2         15.740741
        3         13.544669

## Strongest correlations

- **pclass and fare**: r = -0.548. This is the stronger pair according to the absolute off-diagonal correlation criterion and represents a negative linear association.
- **sibsp and parch**: r = 0.415. This is the stronger pair according to the absolute off-diagonal correlation criterion and represents a positive linear association.

## Multivariate chart interpretations

### Survival by sex

The survival-rate chart shows a clear difference in observed survival between male and female passengers. This indicates that sex is an important descriptive variable for explaining variation in the target. The chart describes an association and does not by itself establish causation.

### Survival by passenger class

Survival rates vary across passenger classes, showing that passenger class contains substantial information about the observed survival outcome. The class breakdown also provides context for the relationship between socioeconomic position and survival in this dataset.

### Survival by sex and class

Combining sex and passenger class reveals interactions that are hidden when either variable is considered alone. The survival pattern is therefore more informative when the two categorical variables are examined jointly rather than independently.

### Age and fare scatter plot

The age-versus-fare scatter plot shows how survival varies across two continuous passenger attributes while sex is displayed separately. It provides a multivariate view of the passenger groups and helps identify whether particular combinations of age and fare are associated with different survival outcomes.

### Fare by class and survival

The fare distributions differ substantially across passenger classes, and survival status can be compared within each class. This chart connects fare, class and survival and shows why a single-variable fare comparison would miss important passenger-group structure.

## Standardization sanity check

Age and fare were standardized using the z-score formula `z = (x - mean) / std`. After transformation, both columns have means approximately equal to zero and standard deviations approximately equal to one. This transformation is an EDA sanity check only and is not passed into the modeling pipeline.
