import pandas as pd
from lambeq import Dataset, NumpyModel, QuantumTrainer
from lambeq import AtomicType, IQPAnsatz, RemoveCupsRewriter
from lambeq import SPSAOptimizer
import numpy as np, os, warnings
from jax import numpy as jnp

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

class CustomNumpyModel(NumpyModel):

    from lambeq.backend.tensor import Diagram

    def get_diagram_output(
        self,
        diagrams: list[Diagram]
    ) -> jnp.ndarray | np.ndarray:

        if self.use_jit:

            lambdified_diagrams = [self._get_lambda(d) for d in diagrams]

            if hasattr(self.weights, "filled"):
                self.weights = self.weights.filled()

            raw_results = [diag_f(self.weights) for diag_f in lambdified_diagrams]

            def force_shape_2_2(arr):
                arr = jnp.asarray(arr)
                flat = arr.flatten()
                # padding o trim per avere 4 elementi
                if flat.size < 4:
                    flat = jnp.pad(flat, (0, 4 - flat.size))
                elif flat.size > 4:
                    flat = flat[:4]
                return flat.reshape((2, 2))

            aligned = [force_shape_2_2(r) for r in raw_results]
            return jnp.stack(aligned, axis=0)

# ============================================================
# Evaluation metrics
# ============================================================

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def flatten_output(y_pred):
    """Appiattisce eventuali output multidimensionali."""
    if y_pred.ndim > 2:
        y_pred = y_pred.reshape(y_pred.shape[0], -1)
    return y_pred


def to_class_indices(y):
    """Converte one-hot o logit in indici di classe."""
    if y.ndim > 1:
        return np.argmax(y, axis=1)
    return y

def eval_metrics():
    """Evaluation metrics (NumPy, Torch, JAX compatible)."""

    def accuracy(y_hat, y_true):
        y_hat = flatten_output(y_hat)
        y_hat = to_class_indices(y_hat)
        y_true = to_class_indices(y_true)
        return accuracy_score(y_true, y_hat)

    def precision(y_hat, y_true):
        y_hat = flatten_output(y_hat)
        y_hat = to_class_indices(y_hat)
        y_true = to_class_indices(y_true)
        return precision_score(y_true, y_hat, average='macro', zero_division=0)

    def recall(y_hat, y_true):
        y_hat = flatten_output(y_hat)
        y_hat = to_class_indices(y_hat)
        y_true = to_class_indices(y_true)
        return recall_score(y_true, y_hat, average='macro', zero_division=0)

    def f1(y_hat, y_true):
        y_hat = flatten_output(y_hat)
        y_hat = to_class_indices(y_hat)
        y_true = to_class_indices(y_true)
        return f1_score(y_true, y_hat, average='macro', zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1-score": f1
    }
# ============================================================
# Load & Split Dataset
# ============================================================

from models.classic.classic_methods_pt1 import load_dataset, split_dataset

# ============================================================
# Parser & Circuit Generator
# ============================================================

from models.classic.classic_methods_pt1 import generate_diagrams

def generate_circuits(diagrams, labels, limit_circuit_qbit = False):
    """
    Generate valid IQP circuits and filter out invalid ones.
    """

    print("\nGenerating circuits using IQPAnsatz...")

    ansatz = IQPAnsatz(
        {
            AtomicType.NOUN: 1,
            AtomicType.SENTENCE: 2
        },
        discard=True,
        n_layers=1,
        n_single_qubit_params=3
    )

    valid_circuits, valid_labels = [], []
    remove_cups = RemoveCupsRewriter()

    for diagram, label in zip(diagrams, labels):
        try:
            circuit = ansatz(remove_cups(diagram))
            n_qubits = circuit.to_tk().n_qubits

            if limit_circuit_qbit and n_qubits > 28:
                print(f"Skipping circuit (too many qubits: {n_qubits})")
                continue

            valid_circuits.append(circuit)
            valid_labels.append(label)

        except KeyError as e:
            print(f"Skipping {diagram} (missing type): {e}")
        except Exception as e:
            print(f"Skipping {diagram} (other error): {e}")

    print(f"{len(valid_circuits)} circuits generated successfully "
          f"out of {len(diagrams)} total.")

    # valid_circuits[0].draw(figsize=(9, 9))

    return valid_circuits, valid_labels

# ============================================================
# Training & Evaluation
# ============================================================

def encode_labels(label_lists):

    all_labels = set()
    for labels in label_lists:
        if isinstance(labels, str):
            labels = [labels]
        all_labels.update(labels)
    print(f"\nEncoding labels {sorted(all_labels)}")

    class_names = sorted(all_labels)  # list of dataset class
    label_map = {name: i for i, name in enumerate(class_names)}

    encoded = []
    for labels in label_lists:
        vec = [0.0] * len(class_names)
        if isinstance(labels, str):
            labels = [labels]
        for l in labels:
            vec[label_map[l]] = 1.0
        encoded.append(vec)

    return encoded

def cross_entropy_loss(logits, labels):
    # Flatten output
    logits = flatten_output(logits)
    labels = flatten_output(labels)

    # Converti da one-hot a indici
    labels = to_class_indices(labels)

    # Log-softmax numericamente stabile
    log_probs = logits - jnp.max(logits, axis=1, keepdims=True)
    log_probs = log_probs - jnp.log(jnp.sum(jnp.exp(log_probs), axis=1, keepdims=True))

    # Loss = -log prob della classe corretta
    loss = -log_probs[jnp.arange(labels.shape[0]), labels]

    return float(jnp.mean(loss))

def run_training(
        train_circuits, val_circuits, test_circuits,
        train_labels, val_labels, test_labels,
        learning_rate, epochs, batch_size,
):
    train_dataset = Dataset(train_circuits, encode_labels(train_labels), batch_size)
    val_dataset = Dataset(val_circuits, encode_labels(val_labels), shuffle=False)

    all_circuits = train_circuits + val_circuits + test_circuits
    model = CustomNumpyModel.from_diagrams(all_circuits, use_jit=True)

    trainer = QuantumTrainer(
        model=model,
        loss_function=cross_entropy_loss,
        epochs=epochs,
        optimizer=SPSAOptimizer,
        optim_hyperparams={
            'a': 0.05,
            'c': 0.06,
            'A': 0.01 * epochs
        },
        evaluate_functions=eval_metrics(),
        evaluate_on_train=True,
        verbose='text',
        seed=42
    )

    print("\nStarting training...")

    trainer.fit(
        train_dataset, val_dataset,
        early_stopping_criterion='accuracy',
        early_stopping_interval=10,
        minimize_criterion=False,
    )

    return trainer, model

# ============================================================
# Plot Metrics
# ============================================================

from models.classic.classic_methods_pt1 import plot_training_metrics

# ============================================================
# Final Evaluation on test
# ============================================================

def evaluate_on_test(model, test_circuits, test_labels):

    metrics = eval_metrics()

    test_acc = metrics["accuracy"](model(test_circuits), test_labels)
    print('Test Accuracy:', test_acc)

    test_rec = metrics["recall"](model(test_circuits), test_labels)
    print('Test Recall:', test_rec)

    test_pre = metrics["precision"](model(test_circuits), test_labels)
    print('Test Precision:', test_pre)

    test_f1 = metrics["f1-score"](model(test_circuits), test_labels)
    print('Test F1:', test_f1)

def evaluate_per_class(model, test_circuits, test_labels):

    class_names = ["O", "PE", "SE", "US"]

    # ---- 1. Ottieni logits dal modello ----
    logits = model(test_circuits)
    logits = np.asarray(logits)

    # ---- 2. Normalizzazione shape (flatten se serve) ----
    logits = flatten_output(logits)

    # ---- 3. Converti logits → classi predette ----
    y_pred = to_class_indices(logits)

    # ---- 4. Etichette vere ----
    y_true = np.asarray(test_labels)
    y_true = to_class_indices(y_true)

    # ---- 5. Report sklearn ----
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    df_report = pd.DataFrame(report_dict).T
    df_report.loc["Average"] = df_report.mean()

    print("\n===== Per-class Metrics (NumPy) =====")
    print(df_report.to_string(float_format="%.3f"))

    return df_report

# ============================================================
# Full Quantum Pipeline
# ============================================================

def run_lambeq_pipeline(
        csv_path, text_col="Requirement", label_col="Type", parser_model = "Bobcat"
):

    # Load dataset
    df = load_dataset(csv_path, text_col, label_col)

    # Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, label_col)

    # Lambeq diagram
    train_diagrams = generate_diagrams(X_train, parser_model)
    val_diagrams = generate_diagrams(X_val, parser_model)
    test_diagrams = generate_diagrams(X_test, parser_model)

    # Lambeq circuit
    train_circuits, train_labels = generate_circuits(train_diagrams, y_train)
    val_circuits, val_labels = generate_circuits(val_diagrams, y_val)
    test_circuits, test_labels = generate_circuits(test_diagrams, y_test)

    # Training model
    trainer, model = run_training(
        train_circuits, val_circuits, test_circuits,
        train_labels, val_labels, test_labels,
        learning_rate=0.05, epochs=50, batch_size=32,
    )

    plot_training_metrics(trainer)
    test_encoded = np.array(encode_labels(test_labels))
    evaluate_on_test(model, test_circuits, test_encoded)
    evaluate_per_class(model, test_circuits, test_encoded)

# ============================================================
# Run the Experiment
# ============================================================

if __name__ == "__main__":

    arta_path = "../../dataset/ARTA/gold/ARTA_Req_balanced.csv"
    pure_path = "../../dataset/ReqExp_PURE/gold/PURE_Req_balanced.csv"
    usor_path = "../../dataset/USoR/gold/USoR_balanced.csv"

    print("\nARTA - Non shot-based quantum model training (Bobcat + Numpy Model + CrossEntropy) ...")
    run_lambeq_pipeline(arta_path, parser_model="Bobcat")

    print("\nPURE - Non shot-based quantum model training (Bobcat + Numpy Model + CrossEntropy) ...")
    run_lambeq_pipeline(pure_path, parser_model="Bobcat")

    print("\nUSoR - Non shot-based quantum model training (Bobcat + Numpy Model + CrossEntropy) ...")
    run_lambeq_pipeline(usor_path, parser_model="Bobcat")

    print("\nARTA - Non shot-based quantum model training (CupsReader + Numpy Model + CrossEntropy) ...")
    run_lambeq_pipeline(arta_path, parser_model="CupsReader")

    print("\nPURE - Non shot-based quantum model training (CupsReader + Numpy Model + CrossEntropy) ...")
    run_lambeq_pipeline(pure_path, parser_model="CupsReader")

    print("\nUSoR - Non shot-based quantum model training (CupsReader + Numpy Model + CrossEntropy) ...")
    run_lambeq_pipeline(usor_path, parser_model="CupsReader")