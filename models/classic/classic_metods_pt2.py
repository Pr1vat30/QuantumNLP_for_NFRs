from lambeq import Dataset, PytorchModel
import torch, pandas as pd, numpy as np

# ============================================================
# Custom lambeq trainer
# ============================================================

from lambeq.training import PytorchTrainer

class MyPytorchTrainer(PytorchTrainer):
    """
    Custom trainer that converts labels to Long Tensor for CrossEntropyLoss.
    """

    def validation_step(
            self,
            batch: tuple[list, torch.Tensor]
        ) -> tuple[torch.Tensor, float]:
        """
        Perform a validation step compatible with CrossEntropyLoss
        """
        x, y = batch
        with torch.no_grad():
            y_hat = self.model(x)

            if y.ndim > 1 and y.dtype == torch.float32:
                y = torch.argmax(y, dim=1)

            y = y.to(self.device).long()

            loss = self.loss_function(y_hat, y)
        return y_hat, loss.item()

    def training_step(
            self,
            batch: tuple[list, torch.Tensor]
        ) -> tuple[torch.Tensor, float]:
        """
        Perform a training step compatible with CrossEntropyLoss
        """
        x, y = batch

        y_hat = self.model(x)

        if y.ndim > 1 and y.dtype == torch.float32:
            y = torch.argmax(y, dim=1)

        y = y.to(self.device).long()

        loss = self.loss_function(y_hat, y)
        self.train_costs.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return y_hat, loss.item()

# ============================================================
# Evaluation metrics
# ============================================================

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def eval_metrics():
    """
    Evaluation metrics for CrossEntropyLoss (multi-class classification).
    """

    def accuracy(y_hat: torch.Tensor, y_true: torch.Tensor):
        preds = torch.argmax(y_hat, dim=1).detach().cpu().numpy()
        y = y_true.detach().cpu().numpy()
        if y.ndim > 1:  # one-hot → indici
            y = torch.argmax(y_true, dim=1).detach().cpu().numpy()
        return accuracy_score(y, preds)

    def precision(y_hat: torch.Tensor, y_true: torch.Tensor):
        preds = torch.argmax(y_hat, dim=1).detach().cpu().numpy()
        y = y_true.detach().cpu().numpy()
        if y.ndim > 1:
            y = torch.argmax(y_true, dim=1).detach().cpu().numpy()
        return precision_score(y, preds, average='weighted', zero_division=0)

    def recall(y_hat: torch.Tensor, y_true: torch.Tensor):
        preds = torch.argmax(y_hat, dim=1).detach().cpu().numpy()
        y = y_true.detach().cpu().numpy()
        if y.ndim > 1:
            y = torch.argmax(y_true, dim=1).detach().cpu().numpy()
        return recall_score(y, preds, average='weighted', zero_division=0)

    def f1(y_hat: torch.Tensor, y_true: torch.Tensor):
        preds = torch.argmax(y_hat, dim=1).detach().cpu().numpy()
        y = y_true.detach().cpu().numpy()
        if y.ndim > 1:
            y = torch.argmax(y_true, dim=1).detach().cpu().numpy()
        return f1_score(y, preds, average='weighted', zero_division=0)

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
    Encode string labels as integer indices per PyTorch CrossEntropyLoss.
    """
    all_labels = sorted(set(label_lists))
    label_map = {name: i for i, name in enumerate(all_labels)}
    return [label_map[label] for label in label_lists]

def run_training(
        train_circuits, val_circuits, test_circuits,
        train_labels, val_labels, test_labels,
        learning_rate=0.01, epochs=50, batch_size=32,
):
    # 1. Encode label
    train_encoded = encode_labels(train_labels)
    val_encoded = encode_labels(val_labels)

    train_encoded = np.array(train_encoded, dtype=np.int64)
    val_encoded = np.array(val_encoded, dtype=np.int64)

    print(f"\nTrain labels (encoded, first 10): {train_encoded[:10].tolist()}")

    # 2. Dataset lambeq
    train_dataset = Dataset(train_circuits, train_encoded.tolist())
    val_dataset   = Dataset(val_circuits, val_encoded.tolist())

    all_circuits = train_circuits + val_circuits + test_circuits

    # 3. PyTorch model
    model = PytorchModel.from_diagrams(all_circuits)
    loss_fn = torch.nn.CrossEntropyLoss()  # Required y_true to be int64

    # 4. Trainer lambeq
    trainer = MyPytorchTrainer(
        model=model,
        loss_function=loss_fn,
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
        train_dataset,
        val_dataset,
        early_stopping_criterion='accuracy',
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

# ============================================================
# Run the Experiment
# ============================================================

if __name__ == "__main__":

    arta_path = "../../dataset/ARTA/gold/ARTA_Req_balanced.csv"
    pure_path = "../../dataset/ReqExp_PURE/gold/PURE_Req_balanced.csv"

    print("\nARTA - lambeq tensor network model training (Bobcat + CrossEntropy) ...")
    run_lambeq_pipeline(arta_path, parser_model="Bobcat")

    print("\nPURE - lambeq tensor network model training (Bobcat + CrossEntropy) ...")
    run_lambeq_pipeline(pure_path, parser_model="Bobcat")

    # print("\nARTA - lambeq tensor network model training (CupsReader + CrossEntropy) ...")
    # run_lambeq_pipeline(arta_path, parser_model="CupsReader")
    #
    # print("\nPURE - lambeq tensor network model training (CupsReader + CrossEntropy) ...")
    # run_lambeq_pipeline(pure_path, parser_model="CupsReader")
