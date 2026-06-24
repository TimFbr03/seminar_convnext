import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm

from models.ConvNeXt import ConvNeXt

NUM_EPOCHS = 16
BATCH_SIZE = 32
LR         = 1e-4
DATA_DIR   = Path("data")


def make_loader(split: str, shuffle: bool) -> DataLoader:
    data = torch.load(DATA_DIR / f"{split}_dataset.pt", weights_only=True)
    dataset = TensorDataset(data["images"], data["labels"])
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def run_epoch(net, loader, criterion, optimizer, device):
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
    return running_loss / len(loader), correct / total


def train_convnext():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = make_loader("train", shuffle=True)
    val_loader   = make_loader("val",   shuffle=False)
    test_loader  = make_loader("test",  shuffle=False)

    net = ConvNeXt(
        in_chans=3,
        num_classes=2,
        depths=[1, 1],       # 2 Stages → 64→16→8
        dims=[16, 32],       # ~50k Parameter
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=LR, weight_decay=0.05)
    best_val_acc = 0.0

    for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):
        train_loss, train_acc = run_epoch(net, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = run_epoch(net, val_loader,   criterion, None,      device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), DATA_DIR / "best_convnext.pt")

        tqdm.write(
            f"Epoch {epoch+1:>2}/{NUM_EPOCHS} | "
            f"train loss={train_loss:.4f} acc={train_acc*100:.2f}% | "
            f"val   loss={val_loss:.4f} acc={val_acc*100:.2f}%"
        )

    net.load_state_dict(torch.load(DATA_DIR / "best_convnext.pt", weights_only=True))
    test_loss, test_acc = run_epoch(net, test_loader, criterion, None, device)
    print(f"\nTest loss={test_loss:.4f} acc={test_acc*100:.2f}%")
