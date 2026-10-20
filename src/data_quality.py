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


def calculate_water_absorption(dry_mass, wet_mass):
    """
    Calculate water absorption (%) using the physical mass relation:
    WA (%) = ((Wet_Mass - Dry_Mass) / Dry_Mass) * 100
    """
    if dry_mass is None or pd.isna(dry_mass):
        return np.nan
    if dry_mass <= 0:
        return 0.0
    return ((wet_mass - dry_mass) / dry_mass) * 100


def validate_membrane_sample(sample_row, tolerance=1.0):
    """
    Validate physical parameters and mathematical consistency of one membrane sample.
    Returns a dictionary of status checks, calculated values and overall status.
    """
    checks = {}
    sample_id = sample_row.get('Sample_ID', 'Unknown')

    # Helper to safely read values and avoid KeyError on missing columns
    def get_value(key, default=None):
        return sample_row.get(key, default)

    dry_mass = get_value('Dry_Mass_g', 0)
    wet_mass = get_value('Wet_Mass_g', 0)
    porosity = get_value('Porosity_%', 0)
    bulk_density = get_value('Bulk_Density_g_cm3', 0)
    flexural_strength = get_value('Flexural_Strength_MPa', 0)
    permeability = get_value('Permeability_L_m2_h_bar', 0)
    diameter = get_value('Diameter_mm', 0)
    thickness = get_value('Thickness_mm', 0)
    reported_wa = get_value('Water_Absorption_%', 0)

    checks['Dry mass > 0'] = float(dry_mass) > 0
    checks['Wet mass >= dry mass'] = float(wet_mass) >= float(dry_mass)
    checks['Porosity in [0, 100]'] = 0 <= float(porosity) <= 100
    checks['Density > 0'] = float(bulk_density) > 0
    checks['Strength > 0'] = float(flexural_strength) > 0
    checks['Permeability >= 0'] = float(permeability) >= 0
    checks['Diameter > 0'] = float(diameter) > 0
    checks['Thickness > 0'] = float(thickness) > 0

    calc_wa = calculate_water_absorption(float(dry_mass), float(wet_mass))
    abs_diff_wa = abs(calc_wa - float(reported_wa))
    checks['Mass relationship consistent'] = abs_diff_wa <= tolerance

    all_passed = all(checks.values())
    status = 'PASS' if all_passed else 'REVIEW'

    return {
        'Sample_ID': sample_id,
        'Checks': checks,
        'Calculated_WA': calc_wa,
        'Reported_WA': reported_wa,
        'WA_Difference': abs_diff_wa,
        'Status': status
    }


def print_validation_report(validation_result):
    """
    Display a readable validation summary for a sample.
    """
    print(f"Sample: {validation_result['Sample_ID']}")
    print('-' * 35)
    for check, passed in validation_result['Checks'].items():
        icon = '✓' if passed else '⚠'
        print(f"{check:<30} {icon}")
    print('-' * 35)
    print(f"Calculated WA: {validation_result['Calculated_WA']:.2f}%")
    print(f"Reported WA:   {validation_result['Reported_WA']:.2f}%")
    print(f"Difference:    {validation_result['WA_Difference']:.2f}%")
    print(f"Overall status: {validation_result['Status']}")
    print('\n')


def validate_all_samples(df, tolerance=1.0):
    """Return validation results for each sample in a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError('validate_all_samples expects a pandas DataFrame.')

    return [validate_membrane_sample(row, tolerance=tolerance) for _, row in df.iterrows()]


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
    print("This module contains reusable data inspection and validation utilities.")
