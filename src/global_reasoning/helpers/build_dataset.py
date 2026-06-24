from PIL import Image, ImageDraw
import random
import torch
import numpy as np
from tqdm import tqdm

SHAPES = ["circle", "square", "triangle", "diamond"]
COLORS = [
    (220, 50,  50),
    (50,  120, 220),
    (50,  180, 80),
    (200, 150, 30),
    (140, 60,  200),
    (30,  180, 180),
]


def draw_shape(draw: ImageDraw.Draw, shape: str, cx: int, cy: int, r: int, color: tuple):
    bbox = (cx - r, cy - r, cx + r, cy + r)
    if shape == "circle":
        draw.ellipse(bbox, fill=color)
    elif shape == "square":
        draw.rectangle(bbox, fill=color)
    elif shape == "triangle":
        pts = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]
        draw.polygon(pts, fill=color)
    elif shape == "diamond":
        pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
        draw.polygon(pts, fill=color)


def create_image(label: bool, size: int = 128) -> tuple[Image.Image, int]:
    """
    Same/Different task: sind beide Formen identisch (gleicher Typ)?

    Label=1: beide Formen haben den GLEICHEN Typ  (same)
    Label=0: beide Formen haben verschiedene Typen (different)

    Warum CNN strukturell scheitert:
      - Größe, Farbe, Position sind randomisiert und kein valides Signal
      - Die Form links liegt komplett in der linken Bildhälfte
      - Die Form rechts liegt komplett in der rechten Bildhälfte
      - Ein CNN muss BEIDE Hälften gleichzeitig integrieren, um zu entscheiden
        ob es sich um den gleichen Typ handelt — das erfordert globalen Kontext
      - Ein ViT sieht von Anfang an alle Patches; Self-Attention verbindet
        linke und rechte Patches direkt im ersten Layer

    Shortcut-Kontrollen:
      - Farbe ist bei same/different unabhängig gewählt → kein Farb-Shortcut
      - Größe ist bei same/different unabhängig gewählt → kein Größen-Shortcut
      - Position ist zufällig in der jeweiligen Hälfte → kein Positions-Shortcut
    """
    img  = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    half = size // 2

    shape_left = random.choice(SHAPES)
    if label:
        shape_right = shape_left                                          # same
    else:
        shape_right = random.choice([s for s in SHAPES if s != shape_left])  # different

    # Farbe und Größe unabhängig → kein Shortcut möglich
    color_left  = random.choice(COLORS)
    color_right = random.choice(COLORS)
    r_left  = random.randint(10, 22)
    r_right = random.randint(10, 22)

    margin = 6
    x_left  = random.randint(r_left  + margin, half - r_left  - margin)
    x_right = random.randint(half + r_right + margin, size - r_right - margin)
    y_left  = random.randint(r_left  + margin, size - r_left  - margin)
    y_right = random.randint(r_right + margin, size - r_right - margin)

    draw_shape(draw, shape_left,  x_left,  y_left,  r_left,  color_left)
    draw_shape(draw, shape_right, x_right, y_right, r_right, color_right)

    return img, int(label)


def create_dataset(n: int, size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    images, labels = [], []
    for i in tqdm(range(n), desc="Building dataset"):
        label = (i % 2 == 0)   # exakt 50/50 balance
        img, lbl = create_image(label, size=size)
        tensor = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
        images.append(tensor)
        labels.append(lbl)

    images = torch.stack(images)
    labels = torch.tensor(labels)
    perm   = torch.randperm(n)
    return images[perm], labels[perm]


def split_dataset(
    images: torch.Tensor,
    labels: torch.Tensor,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
) -> dict[str, dict[str, torch.Tensor]]:
    n       = len(labels)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    return {
        "train": {"images": images[:n_train],                "labels": labels[:n_train]},
        "val":   {"images": images[n_train:n_train + n_val], "labels": labels[n_train:n_train + n_val]},
        "test":  {"images": images[n_train + n_val:],        "labels": labels[n_train + n_val:]},
    }


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=10_000)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--out",  type=str, default="data")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images, labels = create_dataset(args.n, size=args.size)
    splits = split_dataset(images, labels)

    for split_name, split_data in splits.items():
        path = out_dir / f"{split_name}_dataset.pt"
        torch.save(split_data, path)
        n        = len(split_data["labels"])
        pos_rate = split_data["labels"].float().mean().item()
        print(f"{split_name:>5}: {n} samples | shape={split_data['images'].shape} | label mean={pos_rate:.3f}")
