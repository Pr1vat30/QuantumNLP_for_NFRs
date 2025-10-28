import os
import re
import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download the nltk resources needed
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)


# ============================================================
# Dataset Upload
# ============================================================
def load_dataset(csv_path: str, text_col: str = "Requirement") -> pd.DataFrame:
    """
    Loads a csv dataset and returns a DataFrame with the specified columns.
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV: {os.path.basename(csv_path)} ({len(df)} rows)")
    except Exception as e:
        print(f"Error in file {csv_path}: {e}")

    return df


# ============================================================
# Text normalization
# ============================================================
def normalize_text(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = text.encode("ascii", errors="ignore").decode()
    return text


# ============================================================
# Clean punctuation, tags, numbering
# ============================================================
def clean_sentence(sentence: str) -> str:
    sentence = normalize_text(sentence)

    # removes initial numbers (e.g. "2.3.1")
    sentence = re.sub(r'^\s*\d+(\.\d+)*\s*', '', sentence)
    # remove type (O) tag or [SRS001]
    sentence = re.sub(r'\([A-Z]+\)|\[[^\]]*\]', '', sentence)

    sentence = sentence.lower()
    sentence = re.sub(f"[{re.escape(string.punctuation)}]", "", sentence)
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    return sentence


# ============================================================
# Stopword removal + lemmatization
# ============================================================
def process_tokens(sentence: str) -> str:

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    tokens = nltk.word_tokenize(sentence)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# ============================================================
# Single sentence pipeline
# ============================================================
def process_sentence(sentence: str) -> str:
    sentence = clean_sentence(sentence)
    sentence = process_tokens(sentence)
    return sentence


# ============================================================
# Filter requirements with max n words
# ============================================================
def filter_by_length(df: pd.DataFrame, column: str, max_words: int) -> pd.DataFrame:
    """Keep only rows where 'Requirement' has max `max_words` words."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the dataset.")

    df["Word_Count"] = df[column].apply(lambda x: len(str(x).split()))

    # Average calculation before filtering
    mean_len = df["Word_Count"].mean()
    print(f"\nAverage length before filtering: {mean_len:.2f} words")

    filtered = df[df["Word_Count"] <= max_words].drop(columns=["Word_Count"]).reset_index(drop=True)
    print(f"Requirements remaining after filtering (≤ {max_words} words): {len(filtered)}\n")

    return filtered


# ============================================================
# Full dataset pipeline
# ============================================================
def process_dataset(df: pd.DataFrame, text_col: str, max_words: int) -> pd.DataFrame:
    """
    Cleans and filters the dataset directly, returning only the clean requirement column.
    """
    df[text_col] = df[text_col].apply(process_sentence)
    return filter_by_length(df, text_col, max_words)


# ============================================================
# Saving clean dataset
# ============================================================
def save_dataset(df: pd.DataFrame, output_path: str):
    df.to_csv(output_path, index=False)
    print(f"Clean dataset saved in '{output_path}'\n")
    print(df_clean.head())


# ============================================================

if __name__ == "__main__":

    df = load_dataset("../dataset/ARTA/silver/ARTA_Req_labeled.csv", text_col="Requirement")
    df_clean = process_dataset(df, text_col="Requirement", max_words=15)
    save_dataset(df_clean, "../dataset/ARTA/gold/ARTA_Req_normalized.csv")

    df = load_dataset("../dataset/ReqExp_PURE/silver/PURE_Req_labeled.csv", text_col="Requirement")
    df_clean = process_dataset(df, text_col="Requirement", max_words=15)
    save_dataset(df_clean, "../dataset/ReqExp_PURE/gold/PURE_Req_normalized.csv")

