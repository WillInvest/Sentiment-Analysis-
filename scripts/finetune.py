"""Fine-tune FC networks on cached embeddings.

Trains both separate per-horizon models and single multi-output models.
Uses random search over hyperparameter grid. Saves checkpoints, training
logs, and predictions. Fully resumable via progress.json.

Also trains Ridge regression heads (sklearn RidgeCV) as an additional approach.
Ridge predictions are appended to the finetune predictions parquet.

Usage: python scripts/finetune.py
"""

import json
import sys
import random
from pathlib import Path
from itertools import product

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import RidgeCV
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_config
from utils.progress import check_progress, update_progress


# ─── Model Definitions ───


class SeparateFC(nn.Module):
    """FC network for a single horizon."""

    def __init__(self, input_dim: int, hidden_layers: int, neurons: int, activation: str, dropout: float):
        super().__init__()
        layers = []
        in_dim = input_dim
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "leaky_relu": nn.LeakyReLU}[activation]

        for _ in range(hidden_layers):
            layers.extend([nn.Linear(in_dim, neurons), act_fn(), nn.Dropout(dropout)])
            in_dim = neurons
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MultiOutputFC(nn.Module):
    """FC network predicting all horizons at once."""

    def __init__(self, input_dim: int, n_outputs: int, hidden_layers: int, neurons: int, activation: str, dropout: float):
        super().__init__()
        layers = []
        in_dim = input_dim
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "leaky_relu": nn.LeakyReLU}[activation]

        for _ in range(hidden_layers):
            layers.extend([nn.Linear(in_dim, neurons), act_fn(), nn.Dropout(dropout)])
            in_dim = neurons
        layers.append(nn.Linear(in_dim, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─── Training Utilities ───


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE loss ignoring NaN targets (for multi-output model)."""
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)
    return nn.functional.mse_loss(pred[mask], target[mask])


def sample_hyperparams(config: dict) -> list[dict]:
    """Sample n_configs hyperparameter combinations using random search."""
    grid = config["hyperparameters"]["grid"]
    n = config["hyperparameters"]["n_configs"]
    rng = random.Random(config["seed"])

    configs = []
    for _ in range(n):
        cfg = {k: rng.choice(v) for k, v in grid.items()}
        configs.append(cfg)
    return configs


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn,
    hp: dict,
    config: dict,
    device: str,
) -> tuple[nn.Module, list[dict], float]:
    """Train model with early stopping. Returns model, epoch logs, and best val loss."""
    optimizer_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[hp["optimizer"]]
    optimizer = optimizer_cls(model.parameters(), lr=hp["learning_rate"])

    epochs = config["hyperparameters"]["epochs"]
    patience = config["hyperparameters"]["early_stopping_patience"]

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    logs = []

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = loss_fn(pred, y)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        logs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, logs, best_val_loss


def prepare_data_loaders(
    embeddings: dict,
    split_df: pd.DataFrame,
    horizon_col: str | None,
    horizon_cols: list[str] | None,
    batch_size: int,
) -> DataLoader:
    """Build DataLoader from cached embeddings and split dataframe.

    Iterates over each row (article_id, ticker pair) in split_df.
    Embeddings are keyed by article_id (text is the same regardless of ticker),
    but labels come from each specific row to preserve the correct ticker's return.
    """
    X_list, y_list = [], []

    for _, row in split_df.iterrows():
        aid = row["article_id"]
        if aid not in embeddings:
            continue

        X_list.append(embeddings[aid])

        if horizon_col:
            y_list.append(float(row[horizon_col]))
        elif horizon_cols:
            vals = [float(row[c]) for c in horizon_cols]
            y_list.append(vals)

    if not X_list:
        return None

    X = torch.stack(X_list)
    if horizon_col:
        y = torch.tensor(y_list, dtype=torch.float32)
        # Filter NaN
        valid = ~torch.isnan(y)
        X, y = X[valid], y[valid]
    else:
        y = torch.tensor(y_list, dtype=torch.float32)
        # Keep all — masked loss handles NaN

    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ─── Ridge Regression Utilities ───


def build_embedding_matrix(embeddings: dict, df: pd.DataFrame, horizon_col: str | None, horizon_cols: list[str] | None):
    """Build (X, y) numpy arrays from cached embeddings and split dataframe.

    Returns (X, y, valid_rows) where valid_rows is a subset of df with NaN-free labels.
    """
    X_list, y_list, rows = [], [], []

    for _, row in df.iterrows():
        aid = row["article_id"]
        if aid not in embeddings:
            continue

        if horizon_col:
            val = float(row[horizon_col])
            if np.isnan(val):
                continue
            y_list.append(val)
        else:
            vals = [float(row[c]) for c in horizon_cols]
            # Skip rows with any NaN for multi-output
            if any(np.isnan(v) for v in vals):
                continue
            y_list.append(vals)

        X_list.append(embeddings[aid].numpy())
        rows.append(row)

    if not X_list:
        return None, None, None

    X = np.stack(X_list)
    y = np.array(y_list)
    return X, y, pd.DataFrame(rows)


def train_ridge(X_train: np.ndarray, y_train: np.ndarray, seed: int = 42) -> RidgeCV:
    """Train RidgeCV with a wide alpha grid. Returns fitted model."""
    alphas = [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4]
    model = RidgeCV(alphas=alphas, fit_intercept=True, scoring="neg_mean_squared_error", cv=5)
    model.fit(X_train, y_train)
    return model


def generate_ridge_predictions(config: dict, encoder_models: dict, split_type: str = "test") -> list[dict]:
    """Train Ridge models on each encoder/horizon and generate predictions."""
    horizons = config["horizons"]
    horizon_cols = [f"r_{h}d" for h in horizons]
    all_preds = []

    Path("results/checkpoints/ridge").mkdir(parents=True, exist_ok=True)

    for encoder_name, encoder_cfg in encoder_models.items():
        print(f"\n  Ridge: {encoder_name}")
        emb_path = f"results/embeddings/{encoder_name}_embeddings.pt"
        embeddings = torch.load(emb_path, weights_only=False)

        for window in config["windows"]:
            wname = window["name"]
            train_df = pd.read_parquet(f"data/processed/{wname}_train.parquet")
            # Use val as additional training data for Ridge (no early-stopping needed)
            val_df = pd.read_parquet(f"data/processed/{wname}_val.parquet")
            test_df = pd.read_parquet(f"data/processed/{wname}_{split_type}.parquet")

            # Combine train + val for Ridge fitting
            train_val_df = pd.concat([train_df, val_df], ignore_index=True)

            for horizon in horizons:
                h_col = f"r_{horizon}d"
                ckpt_path = Path(f"results/checkpoints/ridge/{encoder_name}_{horizon}d_{wname}.pkl")

                # Train Ridge
                X_tv, y_tv, _ = build_embedding_matrix(embeddings, train_val_df, h_col, None)
                if X_tv is None or len(X_tv) < 20:
                    print(f"    Skipping {encoder_name} {h_col}: insufficient data ({len(X_tv) if X_tv is not None else 0} samples)")
                    continue

                ridge_model = train_ridge(X_tv, y_tv, seed=config["seed"])
                joblib.dump(ridge_model, ckpt_path)
                print(f"    {encoder_name} {h_col} ({wname}): alpha={ridge_model.alpha_:.4f}, train R²={ridge_model.score(X_tv, y_tv):.4f}")

                # Predict on split_type set
                X_test, y_test, valid_rows = build_embedding_matrix(embeddings, test_df, h_col, None)
                if X_test is None:
                    continue

                preds = ridge_model.predict(X_test)

                for i, (_, row) in enumerate(valid_rows.iterrows()):
                    all_preds.append({
                        "encoder": encoder_name,
                        "approach": "ridge",
                        "horizon": h_col,
                        "window": wname,
                        "article_id": row["article_id"],
                        "ticker": row["ticker"],
                        "prediction": float(preds[i]),
                        "actual": float(row[h_col]),
                    })

    return all_preds


# ─── Main Pipeline ───


def main():
    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    horizons = config["horizons"]
    horizon_cols = [f"r_{h}d" for h in horizons]

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    hp_configs = sample_hyperparams(config)

    # Create output dirs
    for d in ["results/training_logs", "results/checkpoints", "results/predictions"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    encoder_models = {
        name: cfg for name, cfg in config["models"].items()
        if "finetune_encoder" in cfg["roles"]
    }

    for encoder_name, encoder_cfg in encoder_models.items():
        print(f"\n{'='*60}")
        print(f"Encoder: {encoder_name}")
        print(f"{'='*60}")

        emb_path = f"results/embeddings/{encoder_name}_embeddings.pt"
        embeddings = torch.load(emb_path, weights_only=False)
        embedding_dim = encoder_cfg["embedding_dim"]

        for window in config["windows"]:
            wname = window["name"]

            train_df = pd.read_parquet(f"data/processed/{wname}_train.parquet")
            val_df = pd.read_parquet(f"data/processed/{wname}_val.parquet")

            # ─── Approach A: Separate per-horizon ───
            for horizon in horizons:
                h_col = f"r_{horizon}d"
                run_key = f"{encoder_name}_separate_{horizon}d_{wname}"

                if check_progress("finetune", run_key) == "done":
                    print(f"  {run_key} already done. Skipping.")
                    continue

                update_progress("finetune", run_key, "in_progress")
                print(f"\n  Training: {run_key}")

                best_val_loss = float("inf")
                best_hp = None
                best_logs = None

                for hp_idx, hp in enumerate(hp_configs):
                    train_loader = prepare_data_loaders(embeddings, train_df, h_col, None, hp["batch_size"])
                    val_loader = prepare_data_loaders(embeddings, val_df, h_col, None, hp["batch_size"])

                    if train_loader is None or val_loader is None:
                        continue

                    model = SeparateFC(
                        input_dim=embedding_dim,
                        hidden_layers=hp["hidden_layers"],
                        neurons=hp["neurons"],
                        activation=hp["activation"],
                        dropout=hp["dropout"],
                    ).to(device)

                    model, logs, epoch_best_val = train_model(
                        model, train_loader, val_loader,
                        nn.MSELoss(), hp, config, device,
                    )

                    if epoch_best_val < best_val_loss:
                        best_val_loss = epoch_best_val
                        best_hp = hp
                        best_logs = logs
                        torch.save(model.state_dict(), f"results/checkpoints/{run_key}_best.pt")

                # Save best training log
                log_data = {"hyperparameters": best_hp, "logs": best_logs, "best_val_loss": best_val_loss}
                Path(f"results/training_logs/{run_key}_log.json").write_text(json.dumps(log_data, indent=2))

                update_progress("finetune", run_key, "done")

            # ─── Approach B: Single multi-output ───
            run_key = f"{encoder_name}_single_{wname}"

            if check_progress("finetune", run_key) == "done":
                print(f"  {run_key} already done. Skipping.")
                continue

            update_progress("finetune", run_key, "in_progress")
            print(f"\n  Training: {run_key}")

            best_val_loss = float("inf")
            best_hp = None
            best_logs = None

            for hp_idx, hp in enumerate(hp_configs):
                train_loader = prepare_data_loaders(embeddings, train_df, None, horizon_cols, hp["batch_size"])
                val_loader = prepare_data_loaders(embeddings, val_df, None, horizon_cols, hp["batch_size"])

                if train_loader is None or val_loader is None:
                    continue

                model = MultiOutputFC(
                    input_dim=embedding_dim,
                    n_outputs=len(horizons),
                    hidden_layers=hp["hidden_layers"],
                    neurons=hp["neurons"],
                    activation=hp["activation"],
                    dropout=hp["dropout"],
                ).to(device)

                model, logs, epoch_best_val = train_model(
                    model, train_loader, val_loader,
                    masked_mse_loss, hp, config, device,
                )

                if epoch_best_val < best_val_loss:
                    best_val_loss = epoch_best_val
                    best_hp = hp
                    best_logs = logs
                    torch.save(model.state_dict(), f"results/checkpoints/{run_key}_best.pt")

            log_data = {"hyperparameters": best_hp, "logs": best_logs, "best_val_loss": best_val_loss}
            Path(f"results/training_logs/{run_key}_log.json").write_text(json.dumps(log_data, indent=2))

            update_progress("finetune", run_key, "done")

    # ─── Generate predictions on val and test sets ───
    if check_progress("finetune", "predictions") == "done":
        print("Predictions already generated. Skipping.")
    else:
        update_progress("finetune", "predictions", "in_progress")
        print("\n\nGenerating val and test predictions...")
        generate_predictions(config, encoder_models, device, split_type="val")
        generate_predictions(config, encoder_models, device, split_type="test")
        update_progress("finetune", "predictions", "done")
    print("Fine-tuning complete.")


def generate_predictions(config: dict, encoder_models: dict, device: str, split_type: str = "test"):
    """Load best models and generate predictions on val or test sets."""
    horizons = config["horizons"]
    horizon_cols = [f"r_{h}d" for h in horizons]

    all_predictions = []

    for encoder_name, encoder_cfg in encoder_models.items():
        embeddings = torch.load(f"results/embeddings/{encoder_name}_embeddings.pt", weights_only=False)
        embedding_dim = encoder_cfg["embedding_dim"]

        for window in config["windows"]:
            wname = window["name"]
            test_df = pd.read_parquet(f"data/processed/{wname}_{split_type}.parquet")

            # Load best hyperparams for this encoder/window
            # Separate models
            for horizon in horizons:
                run_key = f"{encoder_name}_separate_{horizon}d_{wname}"
                log_path = Path(f"results/training_logs/{run_key}_log.json")
                ckpt_path = Path(f"results/checkpoints/{run_key}_best.pt")

                if not log_path.exists() or not ckpt_path.exists():
                    continue

                log_data = json.loads(log_path.read_text())
                hp = log_data["hyperparameters"]

                model = SeparateFC(
                    input_dim=embedding_dim,
                    hidden_layers=hp["hidden_layers"],
                    neurons=hp["neurons"],
                    activation=hp["activation"],
                    dropout=hp["dropout"],
                ).to(device)
                model.load_state_dict(torch.load(ckpt_path, weights_only=True))
                model.eval()

                for _, row in test_df.iterrows():
                    aid = row["article_id"]
                    if aid not in embeddings:
                        continue
                    with torch.no_grad():
                        pred = model(embeddings[aid].unsqueeze(0).to(device)).item()
                    all_predictions.append({
                        "encoder": encoder_name,
                        "approach": "separate",
                        "horizon": f"r_{horizon}d",
                        "window": wname,
                        "article_id": aid,
                        "ticker": row["ticker"],
                        "prediction": pred,
                        "actual": row[f"r_{horizon}d"],
                    })

            # Single multi-output model
            run_key = f"{encoder_name}_single_{wname}"
            log_path = Path(f"results/training_logs/{run_key}_log.json")
            ckpt_path = Path(f"results/checkpoints/{run_key}_best.pt")

            if not log_path.exists() or not ckpt_path.exists():
                continue

            log_data = json.loads(log_path.read_text())
            hp = log_data["hyperparameters"]

            model = MultiOutputFC(
                input_dim=embedding_dim,
                n_outputs=len(horizons),
                hidden_layers=hp["hidden_layers"],
                neurons=hp["neurons"],
                activation=hp["activation"],
                dropout=hp["dropout"],
            ).to(device)
            model.load_state_dict(torch.load(ckpt_path, weights_only=True))
            model.eval()

            for _, row in test_df.iterrows():
                aid = row["article_id"]
                if aid not in embeddings:
                    continue
                with torch.no_grad():
                    preds = model(embeddings[aid].unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
                for h_idx, horizon in enumerate(horizons):
                    all_predictions.append({
                        "encoder": encoder_name,
                        "approach": "single",
                        "horizon": f"r_{horizon}d",
                        "window": wname,
                        "article_id": aid,
                        "ticker": row["ticker"],
                        "prediction": float(preds[h_idx]),
                        "actual": row[f"r_{horizon}d"],
                    })

    pred_df = pd.DataFrame(all_predictions)
    pred_df.to_parquet(f"results/predictions/finetune_{split_type}_predictions.parquet", index=False)
    print(f"Saved {len(pred_df)} {split_type} predictions")


if __name__ == "__main__":
    main()
