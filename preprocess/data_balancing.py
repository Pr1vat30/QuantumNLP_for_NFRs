import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Load and filter dataset
# ============================================================
def load_and_filter_csv(csv_path: str) -> pd.DataFrame:
    """
    Load a CSV file and remove rows where 'Prediction' == 'FR'.
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    if "Prediction" not in df.columns:
        raise ValueError("Column 'Prediction' not found in the dataset.")

    filtered_df = df[df["Prediction"].astype(str).str.strip().ne("F")]
    print(f"Remaining rows after filtering 'FR': {len(filtered_df)}")
    return filtered_df


# ============================================================
# Dataset info
# ============================================================
def show_dataset_info(df: pd.DataFrame):
    """
    Print basic dataset information.
    """
    print("\n=== Dataset Overview ===")
    print(f"Total entries: {len(df)}")

    if "Type" not in df.columns:
        print("Column 'Type' not found in dataset.")
        return

    print("\nClass distribution in 'Type':")
    print(df["Type"].value_counts())


# ============================================================
# Plot class distribution
# ============================================================
def plot_type_distribution(df: pd.DataFrame, title="Relative Frequency of 'Type' Classes"):
    """
    Plot the distribution of classes in the 'Type' column
    including error bars and expected frequency line.
    """
    if "Type" not in df.columns:
        print("Column 'Type' not found in dataset.")
        return

    y = df["Type"].values

    freqs = pd.Series(y).value_counts(normalize=True)
    std_errors = np.sqrt(freqs * (1 - freqs) / len(y))
    expected_frequency = 1 / len(np.unique(y))

    plt.figure(figsize=(8, 5))
    freqs.plot(kind='bar', yerr=std_errors * 1.96, color='steelblue', edgecolor='black', capsize=5)
    plt.axhline(expected_frequency, color='red', linestyle='--', label='Expected frequency (uniform)')
    plt.title(title)
    plt.xlabel("Type")
    plt.ylabel("Relative Frequency")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# ============================================================
# Remove underrepresented classes
# ============================================================
def remove_underrepresented(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove classes in 'Type' whose frequency is below 50% of expected uniform frequency.
    """
    if "Type" not in df.columns:
        raise ValueError("Column 'Type' not found in dataset.")

    y = df["Type"].values
    freqs = pd.Series(y).value_counts(normalize=True)
    expected_frequency = 1 / len(np.unique(y))
    threshold = expected_frequency * 0.75  # changed from 0.5 → 0.75

    print(f"\nExpected frequency: {expected_frequency:.4f}")
    print(f"Removal threshold (75% of expected): {threshold:.4f}")

    valid_classes = freqs[freqs >= threshold].index.tolist()
    removed_classes = freqs[freqs < threshold].index.tolist()

    print(f"\nClasses to remove (below threshold): {removed_classes}")
    print(f"Classes kept: {valid_classes}")

    df_filtered = df[df["Type"].isin(valid_classes)]
    print(f"Remaining rows after removal: {len(df_filtered)}")

    return df_filtered


# ============================================================
# Undersample remaining classes to min count
# ============================================================
def undersample_classes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform undersampling to balance all classes in 'Type' according to the minimum class count.
    """
    if "Type" not in df.columns:
        raise ValueError("Column 'Type' not found in dataset.")

    class_counts = df["Type"].value_counts()
    min_count = class_counts.min()

    print(f"\nPerforming undersampling to {min_count} samples per class...")
    print("Before undersampling:")
    print(class_counts)

    df_balanced = (
        df.groupby("Type", group_keys=False)[df.columns]
        .apply(lambda x: x.sample(n=min_count, random_state=42))
        .reset_index(drop=True)
    )

    print("\nAfter undersampling:")
    print(df_balanced["Type"].value_counts())

    return df_balanced


# ============================================================
# Full pipeline
# ============================================================
def process_dataset(csv_path: str):
    """
    Full pipeline: load, filter, remove underrepresented classes, undersample, analyze, and visualize.
    """
    df_filtered = load_and_filter_csv(csv_path)
    show_dataset_info(df_filtered)
    plot_type_distribution(df_filtered, title="Before Filtering Low-Frequency Classes")

    df_cleaned = remove_underrepresented(df_filtered)
    plot_type_distribution(df_cleaned, title="After Removing Low-Frequency Classes")

    df_balanced = undersample_classes(df_cleaned)
    show_dataset_info(df_balanced)
    plot_type_distribution(df_balanced, title="After Undersampling")

    return df_balanced


# ============================================================

if __name__ == "__main__":
    input_csv = "../dataset/ARTA/gold/ARTA_Req_normalized.csv"
    balanced_df = process_dataset(input_csv)

    output_csv = "../dataset/ARTA/gold/ARTA_Req_balanced.csv"
    balanced_df.to_csv(output_csv, index=False)
    print(f"\nBalanced dataset saved to {output_csv}")

    input_csv = "../dataset/ReqExp_PURE/gold/PURE_Req_normalized.csv"
    balanced_df = process_dataset(input_csv)

    output_csv = "../dataset/ReqExp_PURE/gold/PURE_Req_balanced.csv"
    balanced_df.to_csv(output_csv, index=False)
    print(f"\nBalanced dataset saved to {output_csv}")