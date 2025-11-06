from lambeq import RemoveCupsRewriter, IQPAnsatz, AtomicType, Dataset
from lambeq import Dataset, TketModel, QuantumTrainer, SPSAOptimizer
from pytket.extensions.qiskit import AerBackend
import numpy as np

# ============================================================
# Load & Split Dataset
# ============================================================

from models.classic.classic_methods_pt1 import load_dataset, split_dataset

# ============================================================
# Parser & Circuit Generator
# ============================================================

from models.classic.classic_methods_pt1 import generate_diagrams

def rewrite_diagrams(diagrams, labels):
    """Rewrite and normalize diagrams, filtering invalid ones."""
    remove_cups = RemoveCupsRewriter()

    norm_diagrams, norm_labels = [], []

    for diagram, label in zip(diagrams, labels):
        if diagram is None:
            continue
        try:
            rewritten = remove_cups(diagram)
            normalized = rewritten.normal_form()
            norm_diagrams.append(normalized)
            norm_labels.append(label)
        except Exception as e:
            print(f"Skipping invalid diagram: {e}")
            continue

    return norm_diagrams, norm_labels

from quantum_methods_pt1 import generate_circuits

# ============================================================
# Training & Evaluation
# ============================================================

from quantum_methods_pt1 import encode_labels, eval_metrics, MyCrossEntropyLoss

def run_training(
        train_circuits, val_circuits, test_circuits,
        train_labels, val_labels, test_labels,
        learning_rate, epochs, batch_size,
):
    train_dataset = Dataset(train_circuits, encode_labels(train_labels), batch_size)
    val_dataset = Dataset(val_circuits, encode_labels(val_labels), shuffle=False)

    all_circuits = train_circuits + val_circuits + test_circuits

    backend = AerBackend()
    backend_config = {
        'backend': backend,
        'compilation': backend.default_compilation_pass(2),
        'shots': 4096
    }

    model = TketModel.from_diagrams(all_circuits, backend_config=backend_config)

    loss_fn = MyCrossEntropyLoss()

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

from quantum_methods_pt1 import plot_training_metrics, evaluate_on_test

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

    # Rewrite diagrams
    norm_train_diagram, norm_train_label = rewrite_diagrams(train_diagrams, y_train)
    norm_val_diagram, norm_val_label = rewrite_diagrams(val_diagrams, y_val)
    norm_test_diagram, norm_test_label = rewrite_diagrams(test_diagrams, y_test)

    # Lambeq circuit
    train_circuits, train_labels = generate_circuits(norm_train_diagram, norm_train_label, True)
    val_circuits, val_labels = generate_circuits(norm_val_diagram, norm_val_label, True)
    test_circuits, test_labels = generate_circuits(norm_test_diagram, norm_test_label, True)

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

    # print("\nARTA - Noisy shot-based quantum model training (Bobcat + TketModel Model + CrossEntropy) ...")
    # run_lambeq_pipeline(arta_path, parser_model="Bobcat")

    # print("\nPURE - Noisy shot-based quantum model training (Bobcat + TketModel Model + CrossEntropy) ...")
    # run_lambeq_pipeline(pure_path, parser_model="Bobcat")

    # print("\nARTA - Noisy shot-based quantum model training (CupsReader + TketModel Model + CrossEntropy) ...")
    # run_lambeq_pipeline(arta_path, parser_model="CupsReader")

    print("\nPURE - Noisy shot-based quantum model training (CupsReader + TketModel Model + CrossEntropy) ...")
    run_lambeq_pipeline(pure_path, parser_model="CupsReader")

