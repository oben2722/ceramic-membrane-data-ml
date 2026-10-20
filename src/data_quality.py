import numpy as np
import pandas as pd


def inspect_dataset(dataframe):
    """
    Return a concise summary of dataset structure and quality issues.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("inspect_dataset expects a pandas DataFrame.")

    num_samples = len(dataframe)
    num_variables = dataframe.shape[1]
    num_vars = list(dataframe.select_dtypes(include=[np.number]).columns)
    cat_vars = list(dataframe.select_dtypes(exclude=[np.number]).columns)
    missing_vals = int(dataframe.isnull().sum().sum())
    dup_rows = int(dataframe.duplicated().sum())
    unique_samples = dataframe['Sample_ID'].nunique() if 'Sample_ID' in dataframe.columns else 'N/A'

    print(f"Number of samples: {num_samples}")
    print(f"Number of variables: {num_variables}")
    print(f"Numerical variables ({len(num_vars)}): {num_vars}")
    print(f"Categorical variables ({len(cat_vars)}): {cat_vars}")
    print(f"Total missing values: {missing_vals}")
    print(f"Duplicated rows: {dup_rows}")
    print(f"Unique samples (by Sample_ID): {unique_samples}")


def _safe_display(obj):
    """Display objects in notebooks when available; otherwise print them."""
    try:
        from IPython.display import display
        display(obj)
    except Exception:
        print(obj)


def generate_quality_report(df):
    """
    Generate a data-quality report for an experimental dataset.
    """
    print("=== DATA QUALITY REPORT ===\n")

    # 1. Missing Values
    missing = df.isnull().sum()
    print("--- Missing Values ---")
    if missing.sum() == 0:
        print("No missing values found.")
    else:
        _safe_display(missing[missing > 0])
    print()

    # 2. Duplicate Samples
    duplicates = df.duplicated().sum()
    print("--- Duplicate Rows ---")
    print(f"Found {duplicates} completely duplicated rows.")
    if 'Sample_ID' in df.columns:
        id_duplicates = df['Sample_ID'].duplicated().sum()
        print(f"Found {id_duplicates} duplicated Sample_IDs.")
    print()

    # 3. Data Types and Unique Values
    print("--- Data Types & Unique Values ---")
    summary_df = pd.DataFrame({
        'Data Type': df.dtypes,
        'Unique Values': df.nunique()
    })
    _safe_display(summary_df)
    print()

    # 4. Impossible ranges / invalid values
    print("--- Physical Range Checks ---")
    percentage_cols = [col for col in df.columns if '%' in col]
    out_of_bounds_pct = {}
    for col in percentage_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            invalid = df[(df[col] < 0) | (df[col] > 100)]
            if not invalid.empty:
                out_of_bounds_pct[col] = len(invalid)

    negative_physical_cols = {}
    physical_keywords = ['Mass', 'Temp', 'Time', 'Thickness', 'Diameter', 'Density', 'Strength', 'Permeability']
    for col in df.columns:
        if any(key in col for key in physical_keywords) and pd.api.types.is_numeric_dtype(df[col]):
            invalid = df[df[col] < 0]
            if not invalid.empty:
                negative_physical_cols[col] = len(invalid)

    if not out_of_bounds_pct and not negative_physical_cols:
        print("All percentage and physical properties are within realistic bounds (>= 0, percentages <= 100%).")
    else:
        if out_of_bounds_pct:
            print("Percentage columns with values outside [0, 100]:", out_of_bounds_pct)
        if negative_physical_cols:
            print("Physical columns containing impossible negative values:", negative_physical_cols)
    print()

    # 5. Potential Outliers (using IQR method)
    print("--- Potential Outliers (IQR Method) ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers_found = False
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if not outliers.empty:
            print(f"  * Column '{col}': Found {len(outliers)} potential outlier(s) outside [{lower_bound:.2f}, {upper_bound:.2f}]")
            outliers_found = True
    if not outliers_found:
        print("No statistical outliers detected using the IQR rule.")


if __name__ == "__main__":
    print("This module contains reusable data inspection utilities.")
