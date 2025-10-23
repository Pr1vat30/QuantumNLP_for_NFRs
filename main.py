# ============================================================
# Merge, Convert, and Filter Requirement Files (CSV + XLSX)
# ============================================================

import os
import pandas as pd
from glob import glob
import warnings

# Suppress openpyxl header/footer warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ============================================================
# Load all CSV files from a folder
# ============================================================
def load_csv_from_folder(folder_path: str) -> list[pd.DataFrame]:
    """Load all CSV files from a given folder and return them as a list of DataFrames."""
    csv_files = glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{folder_path}'")

    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"Loaded CSV: {os.path.basename(file)} ({len(df)} rows)")
        except Exception as e:
            print(f"Error in file {file}: {e}")
    return dfs


# ============================================================
# Load all XLSX files, convert to CSV-compatible DataFrames
# ============================================================
def load_xlsx_from_folder(folder_path: str) -> list[pd.DataFrame]:
    """Load all XLSX files, keep only 'Requirement_text' (renamed to 'Requirement')."""
    xlsx_files = glob(os.path.join(folder_path, "*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"No XLSX files found in '{folder_path}'")

    dfs = []
    for file in xlsx_files:
        try:
            df = pd.read_excel(file)
            dfs.append(df)
            print(f"Loaded XLSX: {os.path.basename(file)} ({len(df)} rows)")
        except Exception as e:
            print(f"Error in file {file}: {e}")
    return dfs


# ============================================================
# Merge a list of DataFrames
# ============================================================
def merge_dataframes(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge a list of DataFrames into one DataFrame."""
    if not dfs:
        raise ValueError("Empty DataFrame list — nothing to merge.")
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"Total merged rows: {len(merged_df)}")
    return merged_df


# ============================================================
# Save DataFrame to CSV
# ============================================================
def save_to_csv(df: pd.DataFrame, output_path: str):
    """Save DataFrame to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ File saved to: {output_path}")


# ============================================================
# Filter only rows where Req/Not Req == Req
# ============================================================
def filter_req_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows where 'Req/Not Req' == 'Req' and return only the 'Requirement' column.
    """
    if "Req/Not Req" not in df.columns:
        raise ValueError("Column 'Req/Not Req' not found in the dataset.")
    if "Requirement" not in df.columns:
        raise ValueError("Column 'Requirement' not found in the dataset.")

    # Filter rows where Req/Not Req == 'Req'
    filtered = df[df["Req/Not Req"].astype(str).str.strip().eq("Req")]
    print(f"Rows with Req = 'Req': {len(filtered)}")

    # Return only the 'Requirement' column
    return filtered[["Requirement"]].reset_index(drop=True)


# ============================================================
# Keep only the first column, renamed as 'Requirement'
# ============================================================
def keep_req_only(dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """ For each DataFrame in the list rename the first column to 'Requirement and drop all other columns"""
    cleaned_dfs = []

    for i, df in enumerate(dfs):
        if df.empty:
            print(f"Warning: DataFrame #{i + 1} is empty, skipping.")
            continue

        first_col = df.columns[0]
        df_clean = df[[first_col]].rename(columns={first_col: "Requirement"})
        cleaned_dfs.append(df_clean)
        print(f"Processed DataFrame #{i + 1}: kept column '{first_col}' as 'Requirement'")

    return cleaned_dfs


# ============================================================
# Filter requirements with max n words
# ============================================================
def filter_by_length(df: pd.DataFrame, column: str, min_words: int = 15) -> pd.DataFrame:
    """Keep only rows where 'Requirement' has max `min_words` words."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the dataset.")

    def count_words(text):
        if isinstance(text, str):
            return len(text.split())
        return 0

    df["word_count"] = df[column].apply(count_words)
    filtered = df[df["word_count"] <= min_words].drop(columns=["word_count"])
    print(f"Rows with max {min_words} words: {len(filtered)}")
    return filtered


# ============================================================
# Load, merge, filter, and save the xlsx
# ============================================================
def merge_and_filter_xlsx(input_folder: str, output_file: str, min_words: int = 15):
    """Complete pipeline: load → filter Req → merge → filter by word count → save."""
    dfs = load_xlsx_from_folder(input_folder)
    req_only = keep_req_only(dfs)
    merged = merge_dataframes(req_only)
    filtered = filter_by_length(merged, "Requirement", min_words)
    save_to_csv(filtered, output_file)
    print("\nPreview of final result:")
    print(filtered.head())


# ============================================================
# Load, merge, filter, and save the csv
# ============================================================
def merge_and_filter_csv(input_folder: str, output_file: str, min_words: int = 15):
    """Complete pipeline: load → merge → filter Req → filter by word count → save."""
    dfs = load_csv_from_folder(input_folder)
    merged = merge_dataframes(dfs)
    req_only = filter_req_only(merged)
    filtered = filter_by_length(req_only, "Requirement", min_words)
    save_to_csv(filtered, output_file)
    print("\nPreview of final result:")
    print(filtered.head())


# ============================================================
# Removes functional requirements
# ============================================================
def remove_functional_req(input_csv: str, output_csv: str):
    """Load a CSV, remove all rows where 'Prediction' == 'F', and save the result"""
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")

    # Check for column existence
    if "Prediction" not in df.columns:
        raise ValueError("Column 'Prediction' not found in the dataset.")

    # Filter rows
    filtered_df = df[df["Prediction"].astype(str).str.strip() != "F"]
    print(f"Remaining rows after filtering: {len(filtered_df)}")

    # Save filtered dataset
    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtered dataset saved to {output_csv}")

    return filtered_df

# ============================================================

if __name__ == "__main__":

    input_folder = "dataset/ReqExp_PURE/bronze"
    output_file = "dataset/ReqExp_PURE/silver/requirements.csv"
    merge_and_filter_csv(input_folder, output_file)

    # Example: merge XLSX files first
    input_folder = "dataset/ARTA/bronze"
    output_file = "dataset/ARTA/silver/requirements.csv"
    merge_and_filter_xlsx(input_folder, output_file)

