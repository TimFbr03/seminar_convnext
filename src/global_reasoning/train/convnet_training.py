import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

from models.ConvNet import CNN

NUM_EPOCHS = 8
BATCH_SIZE = 32
LR = 1e-3
DATA_DIR = Path("data")


def make_loader(split: str, shuffle: bool) -> torch.utils.data.DataLoader:
    path = DATA_DIR / f"{split}_dataset.pt"
    data = torch.load(path)
    dataset = torch.utils.data.TensorDataset(data["images"], data["labels"])
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def run_epoch(
    net: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """Run one epoch. If optimizer is None, runs in eval mode (no grad)."""
    training = optimizer is not None
    net.train() if training else net.eval()

    running_loss, correct, total = 0.0, 0, 0

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

    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def train_cnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = make_loader("train", shuffle=True)
    val_loader   = make_loader("val",   shuffle=False)
    test_loader  = make_loader("test",  shuffle=False)

    net = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):
        train_loss, train_acc = run_epoch(net, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = run_epoch(net, val_loader,   criterion, None,      device)

        # Checkpoint best model by val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), DATA_DIR / "best_model.pt")

        tqdm.write(
            f"Epoch {epoch+1:>2}/{NUM_EPOCHS} | "
            f"train loss={train_loss:.4f} acc={train_acc*100:.2f}% | "
            f"val   loss={val_loss:.4f} acc={val_acc*100:.2f}%"
        )

    # # --- Final evaluation on held-out test set ---
    # print("\nLoading best checkpoint for test evaluation...")
    # net.load_state_dict(torch.load(DATA_DIR / "best_model.pt"))
    # test_loss, test_acc = run_epoch(net, test_loader, criterion, None, device)
    # print(f"Test  loss={test_loss:.4f} acc={test_acc*100:.2f}%")
