from lambeq import Dataset, NumpyModel, QuantumTrainer
from lambeq import AtomicType, IQPAnsatz
from lambeq import CrossEntropyLoss, SPSAOptimizer
import numpy as np, tensornetwork as tn
from jax import numpy as jnp


class MyCrossEntropyLoss(CrossEntropyLoss):
    """Custom CrossEntropyLoss per lambeq (compatibile e autonoma)."""
    def __init__(self,
                 use_jax: bool = False,
                 epsilon: float = 1e-9) -> None:
        """Initialise a multiclass cross-entropy loss function.

        Parameters
        ----------
        use_jax : bool, default: False
            Whether to use the Jax variant of numpy.
        epsilon : float, default: 1e-9
            Smoothing constant used to prevent calculating log(0).

        """

        self._epsilon = epsilon

        super().__init__(use_jax)

    def calculate_loss(self,
                       y_pred: np.ndarray | jnp.ndarray,
                       y_true: np.ndarray | jnp.ndarray) -> float:
        """Calculate value of CE loss function."""

        print("\nCalculating loss (+1) ...")

        pred = y_pred.reshape(y_pred.shape[0], -1)
        self._match_shapes(pred, y_true)

        y_pred_smoothed = self._smooth_and_normalise(pred, self._epsilon)

        entropies = y_true * self.backend.log(y_pred_smoothed)
        loss_val: float = -self.backend.sum(entropies) / len(y_true)

        return loss_val


# ============================================================
# Evaluation metrics
# ============================================================

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def eval_metrics():
    """Evaluation metrics (NumPy, Torch, JAX compatible)."""

    def flatten_output(y_pred):
        """Appiattisce eventuali output multidimensionali (es. (batch,2,2) -> (batch,4))."""
        if y_pred.ndim > 2:
            y_pred = y_pred.reshape(y_pred.shape[0], -1)
        return y_pred

    def to_class_indices(y):
        """Converte one-hot o logit in indici di classe."""
        if y.ndim > 1:
            return np.argmax(y, axis=1)
        return y

    def accuracy(y_hat, y_true):
        y_hat = flatten_output(y_hat)
        y_hat = to_class_indices(y_hat)
        y_true = to_class_indices(y_true)
        return accuracy_score(y_true, y_hat)

    def precision(y_hat, y_true):
        y_hat = flatten_output(y_hat)
        y_hat = to_class_indices(y_hat)
        y_true = to_class_indices(y_true)
        return precision_score(y_true, y_hat, average='weighted', zero_division=0)

    def recall(y_hat, y_true):
        y_hat = flatten_output(y_hat)
        y_hat = to_class_indices(y_hat)
        y_true = to_class_indices(y_true)
        return recall_score(y_true, y_hat, average='weighted', zero_division=0)

    def f1(y_hat, y_true):
        y_hat = flatten_output(y_hat)
        y_hat = to_class_indices(y_hat)
        y_true = to_class_indices(y_true)
        return f1_score(y_true, y_hat, average='weighted', zero_division=0)

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

    for diagram, label in zip(diagrams, labels):
        try:
            circuit = ansatz(diagram)
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

def run_training(
        train_circuits, val_circuits, test_circuits,
        train_labels, val_labels, test_labels,
        learning_rate, epochs, batch_size,
):
    train_dataset = Dataset(train_circuits, encode_labels(train_labels), batch_size)
    val_dataset = Dataset(val_circuits, encode_labels(val_labels), shuffle=False)

    all_circuits = train_circuits + val_circuits + test_circuits
    model = NumpyModel.from_diagrams(all_circuits, use_jit=True)

    loss_fn = MyCrossEntropyLoss(use_jax=True)

    trainer = QuantumTrainer(
        model=model,
        loss_function=loss_fn,
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
        train_dataset,
        val_dataset,
        early_stopping_criterion='accuracy',
        early_stopping_interval=10,
        minimize_criterion=False
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

    test_acc = metrics["accuracy"](model.forward(test_circuits), test_labels)
    print('Test Accuracy:', test_acc)

    test_rec = metrics["precision"](model.forward(test_circuits), test_labels)
    print('Test Recall:', test_rec)

    test_pre = metrics["recall"](model.forward(test_circuits), test_labels)
    print('Test Precision:', test_pre)

    test_f1 = metrics["f1-score"](model.forward(test_circuits), test_labels)
    print('Test F1:', test_f1)

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

# ============================================================
# Run the Experiment
# ============================================================

if __name__ == "__main__":

    arta_path = "../../dataset/ARTA/gold/ARTA_Req_balanced.csv"
    pure_path = "../../dataset/ReqExp_PURE/gold/PURE_Req_balanced.csv"

    # print("\nARTA - Non shot-based quantum model training (Bobcat + Numpy Model + CrossEntropy) ...")
    # run_lambeq_pipeline(arta_path, parser_model="Bobcat")

    # print("\nPURE - Non shot-based quantum model training (Bobcat + Numpy Model + CrossEntropy) ...")
    # run_lambeq_pipeline(pure_path, parser_model="Bobcat")

    # print("\nARTA - Non shot-based quantum model training (CupsReader + Numpy Model + CrossEntropy) ...")
    #run_lambeq_pipeline(arta_path, parser_model="CupsReader")

    print("\nPURE - Non shot-based quantum model training (CupsReader + Numpy Model + CrossEntropy) ...")
    run_lambeq_pipeline(pure_path, parser_model="CupsReader")
