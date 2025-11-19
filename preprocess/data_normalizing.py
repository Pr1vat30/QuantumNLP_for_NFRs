import os
import re
import string
import html

import pandas as pd
from tqdm import tqdm

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
import nltk


# ============================================================
# Download risorse NLTK necessarie
# ============================================================
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

# ============================================================
# Dataset Upload
# ============================================================
def load_dataset(csv_path: str, text_col: str = "Requirement") -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV: {os.path.basename(csv_path)} ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return pd.DataFrame(columns=[text_col])

# ============================================================
# Text Normalization
# ============================================================
def normalize_text(text: str) -> str:
    text = html.unescape(text)
    text = (
        text.replace("’", "'")
            .replace("‘", "'")
            .replace("“", '"')
            .replace("”", '"')
    )
    text = text.replace("–", "-").replace("—", "-")
    text = text.encode("ascii", errors="ignore").decode()
    return text

# ============================================================
# Clean and preprocess single sentence
# ============================================================
def clean_sentence(sentence: str) -> str:
    if not isinstance(sentence, str):
        return ""

    # Lowercasing
    sentence = sentence.lower()

    # Normalizzazione caratteri speciali
    sentence = normalize_text(sentence)

    # Rimozione punteggiatura
    sentence = sentence.translate(str.maketrans({p: " " for p in string.punctuation}))

    # Rimozione numeri
    sentence = re.sub(r"\d+", " ", sentence)

    # Pulizia spazi
    sentence = re.sub(r"\s+", " ", sentence).strip()

    return sentence

# ============================================================
# Tokenization, Stopword Removal, Lemmatization
# ============================================================
def process_tokens(sentence: str) -> str:
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    spell = SpellChecker()

    # Tokenizzazione
    tokens = word_tokenize(sentence)

    # Lowercase
    tokens = [t.lower() for t in tokens]

    # Stopword removal
    filtered = [t for t in tokens if t not in stop_words]

    # Filtra eventuali token None
    filtered = [t for t in filtered if t is not None]

    # Lemmatizzazione senza POS
    #lemmatized = [lemmatizer.lemmatize(t) for t in filtered]

    return " ".join(filtered)

# ============================================================
# Pipeline singola frase
# ============================================================
def process_sentence(sentence: str) -> str:
    clean = clean_sentence(sentence)
    processed = process_tokens(clean)
    sentence = re.sub(r"\b[a-zA-Z](?=\b|[^a-zA-Z])", " ", processed)
    return sentence

# ============================================================
# Filtra per lunghezza massima
# ============================================================
def filter_by_length(df: pd.DataFrame, column: str, max_words: int) -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataset.")

    df["Word_Count"] = df[column].apply(lambda x: len(str(x).split()))
    mean_len = df["Word_Count"].mean()
    print(f"\nAverage length before filtering: {mean_len:.2f} words")

    filtered = df[(df["Word_Count"] <= max_words) & (df["Word_Count"] > 2)] \
    .drop(columns=["Word_Count"]) \
    .reset_index(drop=True)
    print(f"Remaining after filtering (≤ {max_words} words): {len(filtered)} rows\n")

    return filtered

# ============================================================
# Full dataset processing (with progress bar)
# ============================================================
def process_dataset(df: pd.DataFrame, text_col: str, max_words: int) -> pd.DataFrame:
    print(f"\nProcessing column '{text_col}'...\n")
    tqdm.pandas(desc="Preprocessing text")
    df[text_col] = df[text_col].astype(str).progress_apply(process_sentence)
    return filter_by_length(df, text_col, max_words)

# ============================================================
# Save dataset
# ============================================================
def save_dataset(df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Clean dataset saved in '{output_path}'\n")
    print(df.head())

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    datasets = [
        {
            "input": "../dataset/ARTA/silver/ARTA_Req_labeled.csv",
            "output": "../dataset/ARTA/gold/ARTA_Req_normalized.csv"
        },
        {
            "input": "../dataset/ReqExp_PURE/silver/PURE_Req_labeled.csv",
            "output": "../dataset/ReqExp_PURE/gold/PURE_Req_normalized.csv"
        },
        {
            "input": "../dataset/USoR/silver/USoR_labeled.csv",
            "output": "../dataset/USoR/gold/USoR_normalized.csv"
        }
    ]

    for data in datasets:
        df = load_dataset(data["input"], text_col="Requirement")
        df_clean = process_dataset(df, text_col="Requirement", max_words=15)
        save_dataset(df_clean, data["output"])