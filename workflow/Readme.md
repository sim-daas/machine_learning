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
























