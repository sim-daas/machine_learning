# MISSING VALUES


## Types of Missing Data
- **MCAR**: Missing Completely at Random - missingness unrelated to any data
- **MAR**: Missing at Random - missingness related to observed data but not missing data itself
- **MNAR**: Missing Not at Random - missingness related to unobserved data

## What To Do When

### For Numerical Data:
1. **When to use Mean/Median Imputation**:
   - Use when data is MCAR
   - Use median when outliers are present
   - Simple and fast but reduces variability

2. **When to use KNN Imputation**:
   - Use when relationships between variables matter
   - Good for complex patterns and MAR data
   - More computationally expensive

### For Categorical Data:
1. **When to use Mode Imputation**:
   - Use when data is MCAR and you need a simple solution
   - When preserving overall distribution is important

2. **When to add "Missing" as a category**:
   - Use when missingness itself is informative (MNAR)
   - When you want to preserve the fact that data was missing

3. **When to use Missing Indicator approach**:
   - Use with MAR or MNAR data
   - When missingness pattern itself might be predictive

4. **When to use MICE Algorithm**:
   - For complex datasets with multiple variable types
   - When you need the most statistically sound approach
   - When dealing with MAR data

## Decision Framework Based on Missing Data Percentage
- **< 5% missing**: Consider complete case analysis or simple imputation
- **5-20% missing**: Use method based on missing data type (MCAR/MAR/MNAR)
- **> 20% missing**: Consider if the variable should be kept or use advanced methods

## Decision Framework Based on Missing Data Type
- **MCAR**: Complete case analysis, mean/median/mode imputation
- **MAR**: KNN, MICE, or regression-based imputation
- **MNAR**: Add missing category, missing indicator, or domain-specific solutions




# SCALING


1. **Min-Max Scaling**: Rescales features to a fixed range, typically [0,1]
   - Best when: You need values within a specific bounded range
   - Sensitive to outliers

2. **Mean Scaling**: Scales values based on mean, typically with range [-1,1]
   - Best when: You want to center around the mean
   - Still sensitive to outliers

3. **Max Absolute Scaling**: Scales by the maximum absolute value
   - Best when: Dealing with sparse data
   - Preserves zero entries in sparse data

4. **Robust Scaling**: Uses median and interquartile range
   - Best when: Your data contains outliers
   - More robust to outliers than other methods

## Types of Feature Scaling

1. **Standardization (Z-score normalization)**:
   - Transforms to mean=0, standard deviation=1
   - Best when: Features follow normal distribution
   - Not bounded to a specific range

2. **Normalization (Min-Max Scaling)**:
   - Rescales features to fixed range [0,1]
   - Best when: Bounded range needed
   - Preserves relationships among data points

3. **Robust Scaling**:
   - Uses statistics less sensitive to outliers
   - Best when: Dataset contains outliers
   - Preserves distribution shape while reducing outlier influence



# Feature Transformation


## Power Transformers

1. **Box-Cox Transformation**:
   - When to use: For positive data with skewed distributions
   - Requires: Strictly positive values (x > 0)
   - Effect: Makes data more normally distributed
   - Best for: Linear regression models, when normality assumption is important

2. **Yeo-Johnson Transformation**:
   - When to use: For data with both positive and negative values
   - Advantage: Handles zero and negative values (unlike Box-Cox)
   - Effect: Makes data more normally distributed
   - Best for: When dealing with mixed-sign data but need normality

3. **Log Transformation**:
   - When to use: For right-skewed data with positive values
   - Effect: Reduces right skewness, compresses high values
   - Best for: Data spanning multiple orders of magnitude

4. **Square Root Transformation**:
   - When to use: For moderately right-skewed, non-negative data
   - Effect: Milder than log transform, stabilizes variance
   - Best for: Count data, moderate skewness

5. **Reciprocal Transformation**:
   - When to use: For severely right-skewed positive data
   - Effect: Inverts relationship, strongest transformation
   - Best for: Extreme skewness

6. **Exponential Transformation**:
   - When to use: For left-skewed data
   - Effect: Increases right skewness
   - Best for: Compressing the lower range of values

7. **Quantile Transformation**:
   - When to use: For arbitrary distributions with outliers
   - Effect: Maps to uniform or normal distribution
   - Best for: When robustness to outliers is critical
   - Preserves: Rank ordering of data












