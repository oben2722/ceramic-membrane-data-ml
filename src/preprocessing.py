"""Preprocessing utilities for ceramic membrane analysis."""

import pandas as pd


def clean_column_names(df):
    """Standardize column names for easier analysis."""
    df = df.copy()
    df.columns = [str(col).strip().replace(" ", "_").lower() for col in df.columns]
    return df


def drop_duplicate_rows(df):
    """Remove fully duplicated rows."""
    return df.drop_duplicates().copy()
