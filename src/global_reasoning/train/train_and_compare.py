"""
Trains CNN and ViT on the same splits and writes a side-by-side comparison.

Metrics collected
-----------------
  - Per-epoch train/val loss and accuracy
  - Total trainable parameters
  - Wall-clock time per epoch (speed)
  - Sample efficiency: val accuracy at 10 / 25 / 50 / 100 % of training data
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from pathlib import Path
from tqdm import tqdm

from models.ConvNet import CNN
from models.ViT import ViT

# ── Config ────────────────────────────────────────────────────────────────────
NUM_EPOCHS   = 8
BATCH_SIZE   = 32
LR           = 1e-3
DATA_DIR     = Path("data")
RESULTS_DIR  = Path("results")
PATIENCE     = 5
SAMPLE_FRACS = [0.10, 0.25, 0.50, 1.00]   # for sample-efficiency sweep

MODELS = {
    "CNN": CNN,
    "ViT": ViT,
}


# ── Data ──────────────────────────────────────────────────────────────────────
def load_split(split: str) -> TensorDataset:
    data = torch.load(DATA_DIR / f"{split}_dataset.pt", weights_only=True)
    return TensorDataset(data["images"], data["labels"])


def make_loader(dataset, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


# ── Training ──────────────────────────────────────────────────────────────────
def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def run_epoch(
    net: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float, float]:
    """Returns (avg_loss, accuracy, wall_seconds)."""
    training = optimizer is not None
    net.train() if training else net.eval()

    running_loss, correct, total = 0.0, 0, 0
    t0 = time.perf_counter()

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    elapsed = time.perf_counter() - t0
    return running_loss / len(loader), correct / total, elapsed


def train_model(
    name: str,
    ModelClass,
    train_ds: TensorDataset,
    val_ds:   TensorDataset,
    device:   torch.device,
) -> dict:
    """Full training run. Returns collected metrics."""
    net       = ModelClass().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    n_params  = count_params(net)

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader   = make_loader(val_ds,   shuffle=False)

    history = []          # list of dicts, one per epoch
    best_val_acc   = 0.0
    epochs_no_improve = 0
    best_ckpt = RESULTS_DIR / f"best_{name}.pt"

    print(f"\n{'─'*60}")
    print(f"  Training {name}  |  params: {n_params:,}")
    print(f"{'─'*60}")

    for epoch in tqdm(range(NUM_EPOCHS), desc=name):
        tr_loss, tr_acc, tr_time = run_epoch(net, train_loader, criterion, optimizer, device)
        va_loss, va_acc, _       = run_epoch(net, val_loader,   criterion, None,      device)

        history.append({
            "epoch":      epoch + 1,
            "train_loss": tr_loss,
            "train_acc":  tr_acc,
            "val_loss":   va_loss,
            "val_acc":    va_acc,
            "epoch_secs": tr_time,
        })

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            epochs_no_improve = 0
            torch.save(net.state_dict(), best_ckpt)
        else:
            epochs_no_improve += 1

        tqdm.write(
            f"  Epoch {epoch+1:>2}/{NUM_EPOCHS} | "
            f"train loss={tr_loss:.4f} acc={tr_acc*100:.2f}% | "
            f"val loss={va_loss:.4f} acc={va_acc*100:.2f}% | "
            f"{tr_time:.1f}s"
        )

        if epochs_no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch+1}")
            break

    return {"name": name, "n_params": n_params, "history": history, "ckpt": best_ckpt}


# ── Sample efficiency ─────────────────────────────────────────────────────────
def sample_efficiency(
    name: str,
    ModelClass,
    train_ds: TensorDataset,
    val_ds:   TensorDataset,
    device:   torch.device,
) -> dict[float, float]:
    """Train on fractions of training data; return {frac: best_val_acc}."""
    val_loader = make_loader(val_ds, shuffle=False)
    results    = {}

    print(f"\n  Sample-efficiency sweep for {name}")
    for frac in SAMPLE_FRACS:
        n_sub  = max(1, int(len(train_ds) * frac))
        subset = Subset(train_ds, list(range(n_sub)))
        loader = make_loader(subset, shuffle=True)

        net       = ModelClass().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=LR)

        best = 0.0
        for _ in range(NUM_EPOCHS):
            run_epoch(net, loader, criterion, optimizer, device)
            _, va_acc, _ = run_epoch(net, val_loader, criterion, None, device)
            best = max(best, va_acc)

        results[frac] = best
        print(f"    {frac*100:>5.0f}% of train ({n_sub:>5} samples) → val acc {best*100:.2f}%")

    return results


# ── Report ────────────────────────────────────────────────────────────────────
def print_report(results: list[dict], sample_eff: dict[str, dict], test_accs: dict[str, float]):
    sep = "═" * 70
    print(f"\n{sep}")
    print("  COMPARISON REPORT")
    print(sep)

    # --- Parameter counts ---
    print("\n  PARAMETERS")
    for r in results:
        print(f"    {r['name']:<6}: {r['n_params']:>10,}")

    # --- Final metrics ---
    print("\n  FINAL EPOCH METRICS")
    header = f"  {'Model':<6} {'Train loss':>10} {'Train acc':>10} {'Val loss':>10} {'Val acc':>10} {'Test acc':>10} {'Avg s/epoch':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        h     = r["history"][-1]
        avg_s = sum(e["epoch_secs"] for e in r["history"]) / len(r["history"])
        print(
            f"  {r['name']:<6} "
            f"{h['train_loss']:>10.4f} "
            f"{h['train_acc']*100:>9.2f}% "
            f"{h['val_loss']:>10.4f} "
            f"{h['val_acc']*100:>9.2f}% "
            f"{test_accs[r['name']]*100:>9.2f}% "
            f"{avg_s:>11.1f}s"
        )

    # --- Sample efficiency ---
    print("\n  SAMPLE EFFICIENCY  (best val acc per training-data fraction)")
    fracs = SAMPLE_FRACS
    header2 = f"  {'Model':<6}" + "".join(f"  {int(f*100):>5}%" for f in fracs)
    print(header2)
    print("  " + "-" * (len(header2) - 2))
    for r in results:
        row = f"  {r['name']:<6}"
        for f in fracs:
            acc = sample_eff[r["name"]][f]
            row += f"  {acc*100:>5.1f}%"
        print(row)

    # --- Epoch-by-epoch val accuracy ---
    print("\n  VAL ACCURACY PER EPOCH")
    max_ep = max(len(r["history"]) for r in results)
    header3 = f"  {'Epoch':>6}" + "".join(f"  {r['name']:>8}" for r in results)
    print(header3)
    print("  " + "-" * (len(header3) - 2))
    for ep in range(1, max_ep + 1):
        row = f"  {ep:>6}"
        for r in results:
            if ep <= len(r["history"]):
                acc = r["history"][ep - 1]["val_acc"]
                row += f"  {acc*100:>7.2f}%"
            else:
                row += f"  {'(stopped)':>8}"
        print(row)

    print(f"\n{sep}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def compare():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = load_split("train")
    val_ds   = load_split("val")
    test_ds  = load_split("test")
    test_loader = make_loader(test_ds, shuffle=False)

    all_results = []
    sample_eff  = {}
    test_accs   = {}

    for name, ModelClass in MODELS.items():
        # Full training run
        result = train_model(name, ModelClass, train_ds, val_ds, device)
        all_results.append(result)

        # Test accuracy using best checkpoint
        net = ModelClass().to(device)
        net.load_state_dict(torch.load(result["ckpt"], weights_only=True))
        _, acc, _ = run_epoch(net, test_loader, nn.CrossEntropyLoss(), None, device)
        test_accs[name] = acc
        print(f"  {name} test accuracy: {acc*100:.2f}%")

        # Sample efficiency sweep
        sample_eff[name] = sample_efficiency(name, ModelClass, train_ds, val_ds, device)

    print_report(all_results, sample_eff, test_accs)


if __name__ == "__main__":
    compare()
