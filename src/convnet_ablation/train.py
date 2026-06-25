import csv
import os
import uuid

from datetime import datetime

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tqdm import tqdm

from model import FashionConvNeXt
from config import experiments

RUN_ID = str(uuid.uuid4())

RESULTS_FILE = f"results/{RUN_ID}.csv"

if not os.path.exists(RESULTS_FILE):

    with open(RESULTS_FILE, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            "run_id",
            "experiment",
            "epoch",
            "train_loss",
            "test_accuracy"
        ])

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def evaluate(model, loader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()

            total += labels.size(0)

    return 100 * correct / total

def run_experiment(
    experiment_name,
    config,
    train_loader,
    test_loader,
    device
):
    print(f'{RUN_ID}')
    print(f"\n{'='*60}")
    print(f"Running: {experiment_name}")
    print(f"{'='*60}")

    model = FashionConvNeXt(**config).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.05
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=30
    )

    best_acc = 0.0

    for epoch in range(30):
        timestamp = datetime.now().isoformat()

        model.train()

        running_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"{experiment_name} Epoch {epoch+1}/30",
            leave=False
        )

        for images, labels in pbar:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        scheduler.step()

        train_loss = running_loss / len(train_loader)

        accuracy = evaluate(
            model,
            test_loader,
            device
        )

        best_acc = max(best_acc, accuracy)

        print(
            f"{experiment_name:<25} "
            f"Epoch {epoch+1:02d} | "
            f"Loss: {train_loss:.4f} | "
            f"Acc: {accuracy:.2f}%"
        )

        with open(
            RESULTS_FILE,
            "a",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                timestamp,
                RUN_ID,
                experiment_name,
                epoch + 1,
                round(train_loss, 4),
                round(accuracy, 2)
            ])

    print(
        f"Best Accuracy ({experiment_name}): "
        f"{best_acc:.2f}%"
    )

    return best_acc

def main():

    device = get_device()

    transform = transforms.ToTensor()

    train_dataset = datasets.STL10(
        root="./data",
        split="train",
        download=True,
        transform=transform
    )

    test_dataset = datasets.STL10(
        root="./data",
        split="test",
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False
    )

    summary = []

    for experiment_name, config in experiments.items():

        best_acc = run_experiment(
            experiment_name,
            config,
            train_loader,
            test_loader,
            device
        )

        summary.append(
            (
                experiment_name,
                best_acc
            )
        )

    print("\nFinal Results")

    for name, acc in summary:

        print(
            f"{name:<25}"
            f"{acc:.2f}%"
        )


if __name__ == "__main__":
    main()
