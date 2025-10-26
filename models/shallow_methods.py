# ============================================================
# Experiment: Compare Embeddings + LazyPredict Classifiers
# ============================================================

import torch, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertModel
import gensim.downloader as api

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# Load Dataset
# ============================================================
def load_dataset(csv_path: str, text_col: str = "Requirement", label_col: str = "Type"):
    df = pd.read_csv(csv_path)
    df = df[[text_col, label_col]].dropna()
    print(f"Loaded {len(df)} rows from {csv_path}")
    return df


# ============================================================
# Split Dataset
# ============================================================
def split_dataset(df: pd.DataFrame, label_col: str = "Type"):
    X_train, X_temp, y_train, y_temp = train_test_split(
        df["Requirement"], df[label_col], test_size=0.3, random_state=42, stratify=df[label_col]
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# Embedding Functions
# ============================================================

# 1️⃣ TF-IDF
def embed_tfidf(X_train, X_val, X_test):
    vectorizer = TfidfVectorizer(max_features=5000)

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)


    X_train_vec = X_train_vec.toarray()
    X_val_vec = X_val_vec.toarray()
    X_test_vec = X_test_vec.toarray()

    return X_train_vec, X_val_vec, X_test_vec

# 2️⃣ Word2Vec
def embed_word2vec(X_train, X_val, X_test):

    w2v = api.load("word2vec-google-news-300")

    def vectorize(texts):
        vecs = []
        for text in texts:
            words = text.lower().split()
            valid_words = [w for w in words if w in w2v]
            if valid_words:
                vec = np.mean(w2v[valid_words], axis=0)
            else:
                vec = np.zeros(w2v.vector_size)
            vecs.append(vec)
        return np.array(vecs)

    print("Generating Word2Vec embeddings...")
    X_train_vec = vectorize(X_train)
    X_val_vec = vectorize(X_val)
    X_test_vec = vectorize(X_test)

    print("Done - dimensione embedding:", w2v.vector_size)
    return X_train_vec, X_val_vec, X_test_vec

# 3️⃣ GloVe
def embed_glove(X_train, X_val, X_test):

    glove = api.load("glove-wiki-gigaword-300")

    def vectorize(texts):
        vecs = []
        for text in texts:
            words = text.lower().split()
            valid_words = [glove[w] for w in words if w in glove]
            if valid_words:
                vec = np.mean(valid_words, axis=0)
            else:
                vec = np.zeros(glove.vector_size)
            vecs.append(vec)
        return np.array(vecs)

    print("Generazione embeddings GloVe...")
    X_train_vec = vectorize(X_train)
    X_val_vec   = vectorize(X_val)
    X_test_vec  = vectorize(X_test)

    print("Embeddings GloVe generati correttamente.")
    return X_train_vec, X_val_vec, X_test_vec

# 4️⃣ FastText
def embed_fasttext(X_train, X_val, X_test):

    ft = api.load("fasttext-wiki-news-subwords-300")

    def vectorize(texts):
        vecs = []
        for text in texts:
            words = text.lower().split()
            valid_words = [w for w in words if w in ft]
            if valid_words:
                vec = np.mean(ft[valid_words], axis=0)
            else:
                vec = np.zeros(ft.vector_size)
            vecs.append(vec)
        return np.array(vecs)

    print("Generazione embeddings FastText...")
    X_train_vec = vectorize(X_train)
    X_val_vec = vectorize(X_val)
    X_test_vec = vectorize(X_test)

    print("Embeddings FastText generati correttamente.")
    return X_train_vec, X_val_vec, X_test_vec

# 5️⃣ BERT (Hugging Face)
def embed_bert(X_train, X_val, X_test, batch_size=32):

    # Carica il tokenizer e il modello
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    # Sposta su GPU se disponibile (opzionale)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    def get_embeddings(texts, batch_size):
        embeddings = []

        # Processa in batch
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            with torch.no_grad():
                # Tokenizza l'intero batch
                encoded = tokenizer.batch_encode_plus(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt',
                    add_special_tokens=True
                )

                # Sposta su device
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                # Estrai embeddings [CLS]
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embeddings)

            # Libera memoria
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        return np.vstack(embeddings)

    train_embeddings = get_embeddings(X_train, batch_size)
    val_embeddings = get_embeddings(X_val, batch_size)
    test_embeddings = get_embeddings(X_test, batch_size)

    return train_embeddings, val_embeddings, test_embeddings


# ============================================================
# LazyPredict Evaluation
# ============================================================
def evaluate_with_lazy(X_train, X_val, y_train, y_val, embedding_name: str):
    print(f"\n=== Evaluating {embedding_name} embeddings ===")
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(X_train, X_val, y_train, y_val)
    model_dictionary = clf.provide_models(X_train, X_val, y_train, y_val)
    print(models.head(5))  # top 5 models by accuracy
    return models, model_dictionary


def report_dataframe(y_true, y_pred, label_names):
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0)
    rows = []
    for label in label_names:
        metrics = report[label]
        rows.append({
            "class": label,
            "precision": round(metrics["precision"],2),
            "recall": round(metrics["recall"],2),
            "f1-score": round(metrics["f1-score"],2)
        })
    avg = report["macro avg"]
    rows.append({
        "class": "Average",
        "precision": round(avg["precision"],2),
        "recall": round(avg["recall"],2),
        "f1-score": round(avg["f1-score"],2)
    })
    return pd.DataFrame(rows)


def evaluate_best_model(X_test, y_test, model_dictionary, results_df, embedding_name: str):
    best_model_name = results_df.sort_values("Accuracy", ascending=False).index[0]
    print(f"\nBest model for {embedding_name}: {best_model_name}")

    model = model_dictionary[best_model_name]

    y_pred = model.predict(X_test)
    df_report = report_dataframe(y_test, y_pred, ["O", "PE", "SE", "US"])

    print(f"📊 Detailed classification report for {embedding_name}:")
    print(df_report.to_string(index=False))

    return df_report


# ============================================================
# Full Experiment Pipeline
# ============================================================
def run_experiment(csv_path: str):
    df = load_dataset(csv_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    results = {}

    embedding_functions = {
        "TF-IDF": embed_tfidf,
        "Word2Vec": embed_word2vec,
        "GloVe": embed_glove,
        "FastText": embed_fasttext,
        # "BERT": embed_bert
    }

    for name, embed_func in embedding_functions.items():
        print(f"\n🔹 Running {name} embedding...")

        # Ottieni gli embedding
        X_train_emb, X_val_emb, X_test_emb = embed_func(X_train, X_val, X_test)

        # Valuta con LazyPredict
        print(f"🔸 Evaluating models for {name}...")
        models, model_dictionary = evaluate_with_lazy(X_train_emb, X_val_emb, y_train_enc, y_val_enc, name)

        evaluate_best_model(X_test_emb, y_test_enc, model_dictionary, models, name)

        results[name] = models

    return results


# =====================================================
# LazyPredict Evaluation
# =====================================================
def display_results(results: dict):
    all_results = []

    table = [
        "Model", "Accuracy", "Balanced Accuracy", "ROC AUC", "F1 Score", "Time Taken", "Embedding"
    ]

    for embed_name, models in results.items():
        if not isinstance(models, pd.DataFrame):
            df_results = pd.DataFrame(models)
        else:
            df_results = models.copy()

        # Aggiungi nome embedding
        df_results["Embedding"] = embed_name

        # Se la colonna "Model" non esiste, aggiungila
        if "Model" not in df_results.columns and df_results.index.name == "Model":
            df_results.reset_index(inplace=True)
        elif "Model" not in df_results.columns and df_results.index.name is None:
            df_results = df_results.reset_index().rename(columns={"index": "Model"})

        # Ordina per metrica principale (Accuracy)
        if "Accuracy" in df_results.columns:
            df_results = df_results.sort_values(by="Accuracy", ascending=False)

        # Stampa risultati completi e leggibili
        print(f"\n📊 Risultati per {embed_name} (ordinati per Accuracy):")
        print(df_results[table].to_string(index=False))

        all_results.append(df_results)


# ============================================================
# Run the Experiment
# ============================================================
if __name__ == "__main__":
    csv_path = "../dataset/ARTA/gold/ARTA_Req_balanced.csv"
    results = run_experiment(csv_path)
    display_results(results)

    csv_path = "../dataset/ReqExp_PURE/gold/PURE_Req_balanced.csv"
    results = run_experiment(csv_path)
    display_results(results)