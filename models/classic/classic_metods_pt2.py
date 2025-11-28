from lambeq import Dataset, PytorchModel
from lambeq.training import PytorchTrainer
import torch, pandas as pd

# ============================================================
# Evaluation metrics
# ============================================================

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def eval_metrics():

    def accuracy(y_hat: torch.Tensor, y_true: torch.Tensor):
        preds = torch.argmax(y_hat, dim=1).numpy()
        y = y_true.numpy()
        return accuracy_score(y, preds)

    def precision(y_hat: torch.Tensor, y_true: torch.Tensor):
        preds = torch.argmax(y_hat, dim=1).numpy()
        y = y_true.numpy()
        return precision_score(y, preds, average='macro', zero_division=0)

    def recall(y_hat: torch.Tensor, y_true: torch.Tensor):
        preds = torch.argmax(y_hat, dim=1).numpy()
        y = y_true.numpy()
        return recall_score(y, preds, average='macro', zero_division=0)

    def f1(y_hat: torch.Tensor, y_true: torch.Tensor):
        preds = torch.argmax(y_hat, dim=1).numpy()
        y = y_true.numpy()
        return f1_score(y, preds, average='macro', zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1-score": f1
    }

# ============================================================
# Load & Split Dataset
# ============================================================

from classic_methods_pt1 import load_dataset, split_dataset

# ============================================================
# Parser & Circuit Generator
# ============================================================

from classic_methods_pt1 import generate_diagrams, generate_circuits

# ============================================================
# Training & Evaluation
# ============================================================

def encode_labels(label_lists):
    """
    Encode string labels as integer indices.
    """
    all_labels = sorted(set(label_lists))
    label_map = {name: i for i, name in enumerate(all_labels)}
    return [label_map[label] for label in label_lists]


def run_training(
        train_circuits, val_circuits, test_circuits,
        train_labels, val_labels, test_labels,
        learning_rate, epochs, batch_size,
):
    # Dataset lambeq
    train_dataset = Dataset(train_circuits, encode_labels(train_labels))
    val_dataset   = Dataset(val_circuits, encode_labels(val_labels))

    all_circuits = train_circuits + val_circuits + test_circuits

    # PyTorch model
    model = PytorchModel.from_diagrams(all_circuits)

    def cross_entropy_loss(logits, labels):
        # Calcola la Cross Entropy (PyTorch gestisce softmax interno)
        return torch.nn.functional.cross_entropy(logits, labels.long())

    # Trainer lambeq
    trainer = PytorchTrainer(
        model=model,
        loss_function=cross_entropy_loss,
        optimizer=torch.optim.AdamW,
        learning_rate=learning_rate,
        epochs=epochs,
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
        minimize_criterion=False
    )

    return trainer, model

# ============================================================
# Plot Metrics
# ============================================================

from classic_methods_pt1 import plot_training_metrics

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


def evaluate_per_class(model, test_circuits, test_labels):

    class_names = ["O", "PE", "SE", "US"]

    # Predizioni
    logits = model(test_circuits)
    y_pred = torch.argmax(logits, dim=1).numpy()

    # Etichette vere (già intere!)
    y_true = torch.as_tensor(test_labels).numpy()

    # Sklearn
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    df_report = pd.DataFrame(report_dict).T
    df_report = df_report[["precision", "recall", "f1-score"]].iloc[:-3]
    df_report.loc["Average"] = df_report.mean()

    print("\n===== Per-class Metrics =====")
    print(df_report.to_string(float_format="%.2f"))

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
        learning_rate=0.01, epochs=50, batch_size=32,
    )

    plot_training_metrics(trainer)

    test_encoded = encode_labels(test_labels)
    test_encoded = torch.tensor(test_encoded, dtype=torch.long)

    evaluate_on_test(model, test_circuits, test_encoded)
    evaluate_per_class(model, test_circuits, test_encoded)

# ============================================================
# Run the Experiment
# ============================================================

if __name__ == "__main__":

    arta_path = "../../dataset/ARTA/gold/ARTA_Req_balanced.csv"
    pure_path = "../../dataset/ReqExp_PURE/gold/PURE_Req_balanced.csv"
    usor_path = "../../dataset/USoR/gold/USoR_balanced.csv"

    print("\nARTA - lambeq tensor network model training (Bobcat + CrossEntropy) ...")
    run_lambeq_pipeline(arta_path, parser_model="Bobcat")

    print("\nPURE - lambeq tensor network model training (Bobcat + CrossEntropy) ...")
    run_lambeq_pipeline(pure_path, parser_model="Bobcat")

    print("\nUSoR - lambeq tensor network model training (Bobcat + CrossEntropy) ...")
    run_lambeq_pipeline(usor_path, parser_model="Bobcat")

    print("\nARTA - lambeq tensor network model training (CupsReader + CrossEntropy) ...")
    run_lambeq_pipeline(arta_path, parser_model="CupsReader")

    print("\nPURE - lambeq tensor network model training (CupsReader + CrossEntropy) ...")
    run_lambeq_pipeline(pure_path, parser_model="CupsReader")

    print("\nUSoR - lambeq tensor network model training (CupsReader + CrossEntropy) ...")
    run_lambeq_pipeline(usor_path, parser_model="CupsReader")
