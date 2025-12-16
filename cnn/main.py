from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split
    from tqdm.auto import tqdm
    from torchvision import datasets, transforms
    from torchvision.models import resnet18, resnet34
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit(
        "Missing Python dependencies for training.\n"
        f"Missing module: {e.name}\n\n"
        "Install with one of:\n"
        "  python -m pip install torch torchvision tqdm\n"
        "  uv pip install torch torchvision tqdm\n"
    ) from e


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TrainConfig:
    data_root: str
    quality: int
    arch: Literal["simple", "resnet18", "resnet34"]
    img_size: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    num_workers: int
    seed: int
    device: str
    amp: bool
    output_dir: str
    train_val_split: float


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /8
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return float(correct) / float(total) if total else 0.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    desc: str = "val",
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for step, (images, targets) in enumerate(pbar, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        total_loss += float(loss.item()) * images.size(0)
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_count += int(images.size(0))

        if step == 1 or step % 25 == 0:
            acc = (total_correct / total_count) if total_count else 0.0
            avg_loss = (total_loss / total_count) if total_count else 0.0
            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.3f}")

    if total_count == 0:
        return 0.0, 0.0
    return total_loss / total_count, total_correct / total_count


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    amp: bool,
    *,
    desc: str = "train",
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for step, (images, targets) in enumerate(pbar, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if amp and scaler is not None and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * images.size(0)
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_count += int(images.size(0))

        if step == 1 or step % 25 == 0:
            acc = (total_correct / total_count) if total_count else 0.0
            avg_loss = (total_loss / total_count) if total_count else 0.0
            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.3f}")

    if total_count == 0:
        return 0.0, 0.0
    return total_loss / total_count, total_correct / total_count


def build_model(arch: Literal["simple", "resnet18", "resnet34"], num_classes: int) -> nn.Module:
    if arch == "simple":
        return SimpleCNN(num_classes=num_classes)
    if arch == "resnet18":
        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch == "resnet34":
        m = resnet34(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise ValueError(f"Unknown arch: {arch}")


def make_transforms(img_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tfms, eval_tfms


def resolve_data_root(repo_root: Path, data_root: str | None, quality: int) -> Path:
    if data_root is not None:
        return Path(data_root).expanduser().resolve()
    # default: <repo>/compressed/q{quality}
    return (repo_root / "compressed" / f"q{quality}").resolve()


def make_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader, DataLoader | None, list[str]]:
    data_root = Path(cfg.data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    train_tfms, eval_tfms = make_transforms(cfg.img_size)

    # Case A (preferred): split folders exist (train/val/test)
    if train_dir.is_dir():
        train_ds = datasets.ImageFolder(train_dir.as_posix(), transform=train_tfms)
        classes = train_ds.classes
    else:
        # Case B: Rust "split" is something else (e.g. raw-img) or user passed the class-root directly.
        # We auto-detect an ImageFolder root:
        # - If data_root itself has class folders -> use it
        # - Else if data_root has exactly one child dir (e.g. raw-img/) and it has class folders -> use that
        candidates: list[Path] = [data_root]
        try:
            children = [p for p in data_root.iterdir() if p.is_dir()]
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing data_root directory: {data_root}") from e

        if len(children) == 1:
            candidates.append(children[0])

        dataset_root: Path | None = None
        for cand in candidates:
            try:
                probe = datasets.ImageFolder(cand.as_posix(), transform=eval_tfms)
            except (FileNotFoundError, RuntimeError):
                continue
            if len(probe.classes) == 0 or len(probe) == 0:
                continue
            dataset_root = cand
            classes = probe.classes
            break

        if dataset_root is None:
            raise FileNotFoundError(
                "Could not find an ImageFolder-compatible dataset root.\n"
                f"Tried: {', '.join(p.as_posix() for p in candidates)}\n\n"
                "Expected one of:\n"
                f"  {data_root}/train/<class>/*.jpg  (preferred)\n"
                f"  {data_root}/<split>/<class>/*.jpg (e.g. raw-img)\n"
                f"  {data_root}/<class>/*.jpg\n"
            )

        full_eval = datasets.ImageFolder(dataset_root.as_posix(), transform=eval_tfms)
        full_train_aug = datasets.ImageFolder(dataset_root.as_posix(), transform=train_tfms)

        n_total = len(full_eval)
        n_val = max(1, int(round(n_total * cfg.train_val_split)))
        n_train = max(1, n_total - n_val)
        if n_train + n_val > n_total:
            n_val = n_total - n_train
        g = torch.Generator().manual_seed(cfg.seed)
        train_subset, val_subset = random_split(full_eval, [n_train, n_val], generator=g)

        train_ds = torch.utils.data.Subset(full_train_aug, train_subset.indices)
        val_ds = val_subset

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(cfg.device.startswith("cuda")),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(cfg.device.startswith("cuda")),
        )

        return train_loader, val_loader, None, classes

    # val: prefer explicit val/ folder; otherwise split train/
    if val_dir.is_dir():
        val_ds = datasets.ImageFolder(val_dir.as_posix(), transform=eval_tfms)
        # keep class ordering consistent if possible
        if val_ds.classes != classes:
            raise RuntimeError(
                "Class folders mismatch between train and val.\n"
                f"train classes: {classes}\n"
                f"val classes:   {val_ds.classes}"
            )
    else:
        full_eval = datasets.ImageFolder(train_dir.as_posix(), transform=eval_tfms)
        n_total = len(full_eval)
        n_val = max(1, int(round(n_total * cfg.train_val_split)))
        n_train = max(1, n_total - n_val)
        if n_train + n_val > n_total:
            n_val = n_total - n_train
        g = torch.Generator().manual_seed(cfg.seed)
        train_subset, val_subset = random_split(full_eval, [n_train, n_val], generator=g)
        # swap transforms: train subset needs augmentation
        full_train_for_split = datasets.ImageFolder(train_dir.as_posix(), transform=train_tfms)
        train_ds = torch.utils.data.Subset(full_train_for_split, train_subset.indices)
        val_ds = val_subset

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.startswith("cuda")),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.startswith("cuda")),
    )

    test_loader: DataLoader | None = None
    if test_dir.is_dir():
        test_ds = datasets.ImageFolder(test_dir.as_posix(), transform=eval_tfms)
        if test_ds.classes != classes:
            raise RuntimeError(
                "Class folders mismatch between train and test.\n"
                f"train classes: {classes}\n"
                f"test classes:  {test_ds.classes}"
            )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(cfg.device.startswith("cuda")),
        )

    return train_loader, val_loader, test_loader, classes


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CNNs on Rust-compressed JPEG dataset: compressed/q{quality}/{split}/{class}/*.jpg"
    )
    parser.add_argument("--data-root", type=str, default=None, help="Path to compressed/q{quality} folder")
    parser.add_argument("--quality", type=int, default=1, choices=[1, 5, 10], help="JPEG quality used in Rust")
    parser.add_argument("--arch", type=str, default="simple", choices=["simple", "resnet18", "resnet34"])
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='e.g. "cpu", "cuda", "cuda:0"',
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP (mixed precision) even on CUDA")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save checkpoints/logs (default: <repo>/runs/<arch>/q{quality}/<timestamp>)",
    )
    parser.add_argument(
        "--train-val-split",
        type=float,
        default=0.1,
        help="Used only if val/ folder doesn't exist; fraction of train used as val",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_root = resolve_data_root(repo_root, args.data_root, args.quality)

    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (repo_root / "runs" / args.arch / f"q{args.quality}" / time.strftime("%Y%m%d-%H%M%S")).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(
        data_root=data_root.as_posix(),
        quality=args.quality,
        arch=args.arch,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        amp=(not args.no_amp),
        output_dir=out_dir.as_posix(),
        train_val_split=args.train_val_split,
    )

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    train_loader, val_loader, test_loader, classes = make_loaders(cfg)
    num_classes = len(classes)

    save_json(out_dir / "config.json", asdict(cfg))
    save_json(out_dir / "classes.json", {"classes": classes})

    model = build_model(cfg.arch, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    best_val_acc = -1.0
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    print(f"Data: {data_root}")
    print(f"Classes ({num_classes}): {classes}")
    print(f"Device: {device} | AMP: {cfg.amp and device.type == 'cuda'}")
    print(f"Output: {out_dir}")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            amp=cfg.amp,
            desc=f"train {epoch}/{cfg.epochs}",
        )
        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            desc=f"val {epoch}/{cfg.epochs}",
        )
        dt = time.time() - t0

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "seconds": dt,
        }
        with (out_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"{dt:.1f}s"
        )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "classes": classes,
            "config": asdict(cfg),
            "val_acc": val_acc,
        }
        torch.save(ckpt, last_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, best_path)

    if test_loader is not None:
        best_ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state"])
        test_loss, test_acc = evaluate(model=model, loader=test_loader, criterion=criterion, device=device)
        save_json(out_dir / "test.json", {"test_loss": test_loss, "test_acc": test_acc})
        print(f"Test | loss {test_loss:.4f} acc {test_acc:.4f}")


if __name__ == "__main__":
    main()
