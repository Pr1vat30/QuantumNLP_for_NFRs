from lambeq import BobcatParser, Dataset, PytorchModel, PytorchTrainer, cups_reader
from lambeq import AtomicType, TensorAnsatz
from lambeq.backend.tensor import Dim
from sympy import default_sort_key
from sklearn.model_selection import train_test_split
import torch, pandas as pd, numpy as np, tensornetwork as tn
import matplotlib.pyplot as plt

"""
Some elements in the original dataset cause the training process to not make progress --> training starts 
but never completes the first epoch. To fix this we manually identify and remove the problematic elements

From ARTA_Req_balanced.csv, the following entry must be removed:
    --> cma report shall returned later 60 second user entered cma report criterion,NFR,PE

From PURE_Req_balanced2.csv, the following entries must be removed:
    --> claus system shall maintain dynamic library data least seven day,NFR,PE
    --> npc sm shall archive security audit data offline minimum two year,NFR,SE

A possible cause of this issue is that the corresponding diagrams and generated circuits for 
these entries are malformed or overly complex, leading the model to stall during training.
"""

to_remove = [
    "cma report shall returned later 60 second user entered cma report criterion",
    "claus system shall maintain dynamic library data least seven day",
    "npc sm shall archive security audit data offline minimum two year"
]

# ============================================================
# Evaluation metrics
# ============================================================

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def single_label_accuracy(y_hat: torch.Tensor, y_true) -> float:
    preds = torch.argmax(y_hat, dim=1).cpu().numpy()
    y = torch.argmax(torch.as_tensor(y_true), dim=1).cpu().numpy()
    return accuracy_score(y, preds)

def single_label_precision(y_hat: torch.Tensor, y_true) -> float:
    preds = torch.argmax(y_hat, dim=1).cpu().numpy()
    y = torch.argmax(torch.as_tensor(y_true), dim=1).cpu().numpy()
    return precision_score(y, preds, average='macro', zero_division=0)

def single_label_recall(y_hat: torch.Tensor, y_true) -> float:
    preds = torch.argmax(y_hat, dim=1).cpu().numpy()
    y = torch.argmax(torch.as_tensor(y_true), dim=1).cpu().numpy()
    return recall_score(y, preds, average='macro', zero_division=0)

def single_label_f1(y_hat: torch.Tensor, y_true) -> float:
    preds = torch.argmax(y_hat, dim=1).cpu().numpy()
    y = torch.argmax(torch.as_tensor(y_true), dim=1).cpu().numpy()
    return f1_score(y, preds, average='macro', zero_division=0)

eval_metrics = {
    "accuracy": single_label_accuracy,
    "precision": single_label_precision,
    "recall": single_label_recall,
    "f1-score": single_label_f1
}

# ============================================================
# Load & Split Dataset
# ============================================================

def load_dataset(csv_path: str, text_col: str, label_col: str):

    df = pd.read_csv(csv_path)
    df = df[[text_col, label_col]].dropna()

    initial_len = len(df)
    df = df[~df[text_col].isin(to_remove)]

    print(f"Removed {initial_len - len(df)} rows matching exclusion list.")
    print(f"Loaded {len(df)} rows from {csv_path}")

    return df

def split_dataset(df: pd.DataFrame, label_col: str):
    X_train, X_temp, y_train, y_temp = train_test_split(
        df["Requirement"], df[label_col], test_size=0.3, random_state=42, stratify=df[label_col]
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train.tolist(), X_val.tolist(), X_test.tolist(), y_train.tolist(), y_val.tolist(), y_test.tolist()

# ============================================================
# Parser & Circuit Generator
# ============================================================

def generate_diagrams(data, parser_model: str = "Bobcat"):

    print(f"\nParsing sentences into diagrams...")

    if parser_model == "Bobcat":
        parser = BobcatParser(verbose='progress')
    elif parser_model == "CupsReader":
        parser = cups_reader
    else: raise ValueError(f"Unknown model '{parser_model}'.")

    diagrams = parser.sentences2diagrams(data)
    print("Parsed sentences successfully.")
    # diagrams[0].draw(figsize=(7, 3))

    return diagrams

def generate_circuits(diagrams, labels):
    print("\nGenerating circuits...")

    ansatz = TensorAnsatz({
        AtomicType.NOUN: Dim(4),
        AtomicType.SENTENCE: Dim(4)
    })

    valid_circuits, valid_labels = [], []

    for diagram, label in zip(diagrams, labels):
        try:

            circuit = ansatz(diagram)

            # test size of tensor circuits
            if circuit.cod != Dim(4):
                print(f"Skipping {circuit} (tensor size={circuit.cod})")
                continue

            # test on tensor circuits
            syms = sorted(circuit.free_symbols, key=default_sort_key)
            sym_dict = {k: torch.ones(k.size) for k in syms}
            subbed_diagram = circuit.lambdify(*syms)(*sym_dict.values())

            # circuits contraction into tensor
            result = subbed_diagram.eval(contractor=tn.contractors.auto)
            if isinstance(result, torch.Tensor):
                result = result.detach().cpu().numpy()

            # check coherency of dtype
            if result.dtype != 'float32':
                print(f"Skipping {diagram} (result dtype={result.dtype})")
                continue

            valid_circuits.append(circuit)
            valid_labels.append(label)

        except KeyError as e:
            print(f"Skipping {diagram} diagram (missing type): {e}")
        except Exception as e:
            print(f"Skipping {diagram} diagram (other error): {e}")

    print(f"{len(valid_circuits)} circuits generated successfully.")
    # train_circuits[0].draw(figsize=(7, 3))

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
    train_dataset = Dataset(train_circuits, encode_labels(train_labels))
    val_dataset = Dataset(val_circuits, encode_labels(val_labels))

    all_circuits = train_circuits + val_circuits + test_circuits
    model = PytorchModel.from_diagrams(all_circuits).to(torch.float32)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    trainer = PytorchTrainer(
        model=model,
        loss_function=loss_fn,
        optimizer=torch.optim.AdamW,
        learning_rate=learning_rate,
        epochs=epochs,
        evaluate_functions=eval_metrics,
        evaluate_on_train=True,
        verbose='text',
        seed=42
    )

    print("\nStarting training...")

    trainer.fit(
        train_dataset, val_dataset, early_stopping_criterion='accuracy',
        early_stopping_interval=10, minimize_criterion=False
    )

    return trainer, model

# ============================================================
# Plot Metrics
# ============================================================

def plot_training_metrics(trainer):

    # --- get the early stop epoch ---
    early_stop_epoch = min(len(trainer.train_epoch_costs), len(trainer.val_costs))
    total_epochs = getattr(trainer, "epochs", early_stop_epoch)

    print(f"\n[Info] Early stopping detected at epoch {early_stop_epoch}/{total_epochs}")

    metrics = list(trainer.train_eval_results.keys())
    main_metric = "accuracy"
    rows = len(metrics) + 1  # loss + metrics
    fig, axes = plt.subplots(rows, 2, sharey='row', figsize=(10, 3 * rows))

    for ax in axes[:, 0]:
        ax.set_title("Training set")
        ax.set_xlabel("Epochs")
    for ax in axes[:, 1]:
        ax.set_title("Validation set")
        ax.set_xlabel("Epochs")

    epochs = np.arange(1, early_stop_epoch + 1)
    colours = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    # ======== loss function ========
    loss_color = next(colours)
    axes[0, 0].plot(epochs, trainer.train_epoch_costs, color=loss_color)
    axes[0, 0].set_ylabel("Loss")
    axes[0, 1].plot(epochs, trainer.val_costs, color=loss_color)
    axes[0, 1].set_ylabel("Loss")

    # ======== other metrics ========
    for i, metric in enumerate(metrics, start=1):
        color = next(colours)
        train_values = trainer.train_eval_results[metric]
        val_values = trainer.val_eval_results[metric]

        axes[i, 0].plot(epochs, train_values, color=color)
        axes[i, 0].set_ylabel(metric.capitalize())
        axes[i, 1].plot(epochs, val_values, color=color)
        axes[i, 1].set_ylabel(metric.capitalize())

    # ======== early stop marker ========
    best_epoch = int(np.argmax(trainer.val_eval_results[main_metric]))
    for ax_row in axes:
        for ax in ax_row:
            ax.plot(
                best_epoch + 1, ax.lines[0].get_ydata()[best_epoch], 'o', color='black', fillstyle='none'
            )

    last_row = metrics.index(main_metric) + 1 if main_metric in metrics else 1
    ax_val_main = axes[last_row, 1]
    y_best = trainer.val_eval_results[main_metric][best_epoch]
    ax_val_main.text(best_epoch + 1.2, y_best, 'early stopping', va='center')

    fig.tight_layout()
    plt.show()

# ============================================================
# Final Evaluation on test
# ============================================================
def evaluate_on_test(model, test_circuits, test_labels):

    test_acc = single_label_accuracy(model.forward(test_circuits), test_labels)
    print('Test Accuracy:', test_acc)

    test_rec = single_label_recall(model.forward(test_circuits), test_labels)
    print('Test Recall:', test_rec)

    test_pre = single_label_precision(model.forward(test_circuits), test_labels)
    print('Test Precision:', test_pre)

    test_f1 = single_label_f1(model.forward(test_circuits), test_labels)
    print('Test F1:', test_f1)

# ============================================================
# Full Classic Pipeline
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
        learning_rate=0.01, epochs=50, batch_size=16,
    )

    plot_training_metrics(trainer)
    evaluate_on_test(model, test_circuits, encode_labels(test_labels))

# ============================================================
# Run the Experiment
# ============================================================

if __name__ == "__main__":

    arta_path = "../../dataset/ARTA/gold/ARTA_Req_balanced.csv"
    pure_path = "../../dataset/ReqExp_PURE/gold/PURE_Req_balanced.csv"

    # print("\nARTA - lambeq tensor network model training (Bobcat + BCELoss) ...")
    # run_lambeq_pipeline(arta_path, parser_model="Bobcat")

    # print("\nPURE - lambeq tensor network model training (Bobcat + BCELoss) ...")
    # run_lambeq_pipeline(pure_path, parser_model="Bobcat")

    print("\nARTA - lambeq tensor network model training (CupsReader + BCELoss) ...")
    run_lambeq_pipeline(arta_path, parser_model="CupsReader")

    print("\nPURE - lambeq tensor network model training (CupsReader + BCELoss) ...")
    run_lambeq_pipeline(pure_path, parser_model="CupsReader")
