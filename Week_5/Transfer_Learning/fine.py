# landcover_train_and_show.py
# Works when you only have: data/Urban, data/Forest, data/Water (no train/val)
# It will create ./data_split/train and ./data_split/val automatically, then train, evaluate,
# and save visual predictions for both val set and any images found in ./uploads/.
#
# This version removes Matplotlib entirely and uses Pillow (PIL) for visualizations.
# It ALSO logs per-epoch accuracy & loss and saves line charts (PIL) and CSV/JSON history.
# It is FULLY OFFLINE: no downloads for pretrained weights (avoids HTTP 403).

import os, json, time, copy, random, csv
from glob import glob
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from PIL import Image, ImageDraw, ImageFont

# =========================
# Config
# =========================
# Adjust RAW_DIR to the folder that contains your class subfolders (Urban/Forest/Water etc.)
RAW_DIR = "./data/data"            # e.g., "./data" if your structure is data/Urban, data/Forest, data/Water
SPLIT_DIR = "./data/data_split"    # Train/Val will be created here
OUTPUTS = "./outputs"              # Models, plots, annotated preds saved here

SPLIT_RATIO = 0.8                  # train/val split on your single folder per class
INPUT_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 3                         # Consider 20–30 if training from scratch for better results
LR = 1e-3                          # Consider 3e-3 if training from scratch
FREEZE_BACKBONE = False            # IMPORTANT: keep False when training from scratch (weights=None)
UNFREEZE_AFTER = None              # e.g., set 4 to unfreeze layer4+fc after 4 epochs (when using pretrained)
NUM_WORKERS = 0
SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# If you have a local ResNet-50 checkpoint (e.g., resnet50-11ad3fa6.pth), set the path here.
# Otherwise, leave as None to train from scratch.
LOCAL_RESNET50: Optional[str] = None
# Example:
# LOCAL_RESNET50 = r"C:\Users\Pratik.Jadhav\.cache\torch\hub\checkpoints\resnet50-11ad3fa6.pth"


# =========================
# Utils
# =========================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def hardlink_or_copy(src, dst):
    try:
        os.link(src, dst)  # hardlink (fast, saves space)
    except Exception:
        import shutil
        shutil.copy(src, dst)

def is_already_split(root: str):
    return os.path.isdir(os.path.join(root, "train")) and os.path.isdir(os.path.join(root, "val"))

def denorm(t, mean, std):
    mean = torch.tensor(mean).view(3,1,1)
    std = torch.tensor(std).view(3,1,1)
    return t * std + mean

def try_load_font(size=14):
    # Try a truetype font for better rendering; fall back to default.
    for fname in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(fname, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


# =========================
# Prepare split if needed
# =========================
def auto_split_from_single_folder(raw_dir: str, split_dir: str, split_ratio: float=0.8):
    """
    Create train/val under split_dir from raw_dir that contains per-class folders only.
    Does NOT modify originals. Uses hardlinks (or copies).
    """
    classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    classes = sorted([c for c in classes if not c.startswith(".")])
    if not classes:
        raise RuntimeError(f"No class folders found under {raw_dir}")

    print("[Prepare] Found classes:", classes)
    for phase in ["train", "val"]:
        for c in classes:
            ensure_dir(os.path.join(split_dir, phase, c))

    exts = ("*.jpg","*.jpeg","*.png","*.tif","*.tiff","*.bmp","*.JPG","*.JPEG","*.PNG")
    for c in classes:
        files = []
        for ext in exts:
            files.extend(glob(os.path.join(raw_dir, c, ext)))
        if not files:
            print(f"[WARN] No images in {os.path.join(raw_dir, c)}")
            continue
        random.shuffle(files)
        n_train = int(len(files) * split_ratio)
        train_files, val_files = files[:n_train], files[n_train:]

        for f in train_files:
            dst = os.path.join(split_dir, "train", c, os.path.basename(f))
            if not os.path.exists(dst):
                hardlink_or_copy(f, dst)
        for f in val_files:
            dst = os.path.join(split_dir, "val", c, os.path.basename(f))
            if not os.path.exists(dst):
                hardlink_or_copy(f, dst)

    print(f"[Prepare] Created split under {split_dir}")


# =========================
# Data
# =========================
def make_dataloaders(data_dir: str, input_size: int, batch_size: int, num_workers: int):
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.15, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    image_datasets = {
        "train": datasets.ImageFolder(os.path.join(data_dir, "train"), tf_train),
        "val": datasets.ImageFolder(os.path.join(data_dir, "val"), tf_eval),
    }
    dataloaders = {
        k: DataLoader(v, batch_size=batch_size, shuffle=(k=="train"),
                      num_workers=num_workers, pin_memory=True)
        for k, v in image_datasets.items()
    }
    sizes = {k: len(v) for k,v in image_datasets.items()}
    class_to_idx = image_datasets["train"].class_to_idx
    idx_to_class = {v: k for k,v in class_to_idx.items()}

    print("[Data] Classes:", [idx_to_class[i] for i in sorted(idx_to_class.keys())])
    print("[Data] Train size:", sizes["train"], "| Val size:", sizes["val"])
    return image_datasets, dataloaders, sizes, idx_to_class


# =========================
# Model (Offline)
# =========================
def create_resnet50(num_classes: int,
                    freeze_backbone: bool = False,
                    local_ckpt: Optional[str] = None):
    """
    Create a ResNet-50 without triggering any internet downloads.
    Optionally load a local checkpoint if provided.
    """
    # Important: weights=None prevents any online download
    model = models.resnet50(weights=None)  # (older torchvision: pretrained=False)

    # Optional: Load a local pretrained weight file if you have it
    if local_ckpt and os.path.isfile(local_ckpt):
        try:
            state = torch.load(local_ckpt, map_location="cpu")
            model.load_state_dict(state, strict=False)
            print(f"[Load] Loaded local weights: {local_ckpt}")
        except Exception as e:
            print(f"[Warn] Failed to load local weights: {e}")
    else:
        if local_ckpt:
            print(f"[Warn] Local weights file not found: {local_ckpt}. Continuing from scratch.")
        else:
            print("[Info] No local weights provided; training from scratch.")

    # Replace classifier head
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)

    # Freezing only makes sense if you actually loaded pretrained features
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True

    return model


# =========================
# Simple PIL Curves Plotter
# =========================
def plot_curves_pil(history: Dict[str, list], out_dir: str):
    """
    Draw simple line plots for train/val loss and accuracy using PIL.
    Saves: curve_loss.png, curve_accuracy.png
    """
    ensure_dir(out_dir)
    font = try_load_font(14)
    font_small = try_load_font(12)

    def _draw_plot(series_list, labels, title, y_label, out_path):
        # Canvas
        W, H = 900, 520
        margin_l, margin_r, margin_t, margin_b = 80, 30, 60, 70
        img = Image.new("RGB", (W, H), (255, 255, 255))
        dr = ImageDraw.Draw(img)

        # Title
        dr.text((20, 15), title, fill=(0,0,0), font=font)

        # Axis box
        x0, y0 = margin_l, H - margin_b
        x1, y1 = W - margin_r, margin_t
        dr.rectangle([x0, y1, x1, y0], outline=(0,0,0), width=1)

        # Prepare data
        max_len = max(len(s) for s in series_list) if series_list else 0
        if max_len == 0:
            img.save(out_path)
            return

        # Determine y scale
        y_min = min(min(s) for s in series_list if len(s) > 0)
        y_max = max(max(s) for s in series_list if len(s) > 0)
        if y_min == y_max:
            y_min -= 0.5
            y_max += 0.5

        def to_xy(idx, val):
            # map data index and value to canvas coords
            if max_len == 1:
                tx = x0
            else:
                tx = x0 + (x1 - x0) * (idx / (max_len - 1))
            # invert y axis (top is small)
            ty = y0 - (y0 - y1) * ((val - y_min) / (y_max - y_min))
            return (tx, ty)

        # Gridlines (Y)
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            yy = y0 - (y0 - y1) * frac
            dr.line([(x0, yy), (x1, yy)], fill=(230,230,230))
            y_val = y_min + (y_max - y_min) * frac
            label = f"{y_val:.3f}"
            tw, th = dr.textsize(label, font=font_small)
            dr.text((x0 - tw - 8, yy - th/2), label, fill=(0,0,0), font=font_small)

        # X ticks (epochs)
        for e in range(max_len):
            xx, _ = to_xy(e, y_min)
            dr.line([(xx, y0), (xx, y0+5)], fill=(0,0,0))
            lab = f"{e+1}"
            tw, th = dr.textsize(lab, font=font_small)
            dr.text((xx - tw/2, y0 + 8), lab, fill=(0,0,0), font=font_small)

        # Draw series
        colors = [(31,119,180), (214,39,40), (44,160,44), (255,127,14)]
        for si, s in enumerate(series_list):
            pts = [to_xy(i, v) for i, v in enumerate(s)]
            col = colors[si % len(colors)]
            for i in range(1, len(pts)):
                dr.line([pts[i-1], pts[i]], fill=col, width=3)
            for (px, py) in pts:
                dr.ellipse([px-3, py-3, px+3, py+3], fill=col)

        # Labels
        dr.text((x0 + (x1-x0)//2 - 40, H - margin_b + 35), "Epoch", fill=(0,0,0), font=font)
        # Y-axis label (rotated)
        ylab_img = Image.new("RGBA", (120, 28), (255,255,255,0))
        d2 = ImageDraw.Draw(ylab_img)
        d2.text((0,0), y_label, fill=(0,0,0), font=font)
        ylab_img = ylab_img.rotate(90, expand=True)
        img.paste(ylab_img, (20, margin_t + (y0 - y1)//2 - ylab_img.height//2), ylab_img)

        # Legend
        lx = x0 + 10
        ly = y1 - 40
        for si, lab in enumerate(labels):
            col = colors[si % len(colors)]
            dr.rectangle([lx, ly, lx+16, ly+16], fill=col)
            dr.text((lx+22, ly-2), lab, fill=(0,0,0), font=font_small)
            ly += 22

        img.save(out_path)
        print(f"[Saved] {out_path}")

    # Build data for plots
    loss_train = history.get("train_loss", [])
    loss_val   = history.get("val_loss", [])
    acc_train  = history.get("train_acc", [])
    acc_val    = history.get("val_acc", [])

    _draw_plot(
        [loss_train, loss_val],
        ["train_loss", "val_loss"],
        "Training/Validation Loss",
        "Loss",
        os.path.join(out_dir, "curve_loss.png")
    )
    _draw_plot(
        [acc_train, acc_val],
        ["train_acc", "val_acc"],
        "Training/Validation Accuracy",
        "Accuracy",
        os.path.join(out_dir, "curve_accuracy.png")
    )

def save_history(history: Dict[str, list], out_dir: str):
    ensure_dir(out_dir)
    # JSON
    with open(os.path.join(out_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    # CSV
    rows = []
    max_len = max(len(history.get("epoch", [])),
                  len(history.get("train_loss", [])),
                  len(history.get("val_loss", [])),
                  len(history.get("train_acc", [])),
                  len(history.get("val_acc", [])))
    for i in range(max_len):
        rows.append({
            "epoch": history.get("epoch", [])[i] if i < len(history.get("epoch", [])) else i+1,
            "train_loss": history.get("train_loss", [None]*max_len)[i],
            "val_loss": history.get("val_loss", [None]*max_len)[i],
            "train_acc": history.get("train_acc", [None]*max_len)[i],
            "val_acc": history.get("val_acc", [None]*max_len)[i],
        })
    with open(os.path.join(out_dir, "train_history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","train_acc","val_acc"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Saved] {os.path.join(out_dir, 'train_history.json')} and train_history.csv")


# =========================
# Train (with history)
# =========================
def train(model, loaders, sizes, device, epochs=8, lr=1e-3, unfreeze_after=None, out_dir="outputs"):
    ensure_dir(out_dir)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_wts = copy.deepcopy(model.state_dict())
    best_acc, no_improve, patience = 0.0, 0, 3
    best_path = os.path.join(out_dir, "best_resnet50.pth")

    # History containers
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}\n" + "-"*22)
        t0 = time.time()

        if unfreeze_after is not None and epoch + 1 == unfreeze_after:
            print("[Train] Unfreezing 'layer4' + 'fc' for fine-tuning...")
            for name, p in model.named_parameters():
                if "layer4" in name or "fc" in name:
                    p.requires_grad = True
            optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr*0.1)

        epoch_stats = {}
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in loaders[phase]:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = outputs.max(1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            if phase == "train":
                scheduler.step()
            epoch_loss = running_loss / sizes[phase] if sizes[phase] > 0 else 0.0
            epoch_acc = running_corrects.double().item() / sizes[phase] if sizes[phase] > 0 else 0.0
            epoch_stats[phase] = (epoch_loss, epoch_acc)
            print(f"{phase} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            if phase == "val":
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_wts, best_path)
                    no_improve = 0
                else:
                    no_improve += 1

        # Log history
        history["epoch"].append(epoch+1)
        history["train_loss"].append(epoch_stats["train"][0])
        history["val_loss"].append(epoch_stats["val"][0])
        history["train_acc"].append(epoch_stats["train"][1])
        history["val_acc"].append(epoch_stats["val"][1])

        print(f"[Epoch Time] {time.time()-t0:.1f}s | Best Val Acc: {best_acc:.4f}")
        if no_improve >= patience:
            print("[Early Stopping] No improvement. Stopping.")
            break

    model.load_state_dict(best_wts)
    torch.save(model.state_dict(), os.path.join(out_dir, "last_epoch_resnet50.pth"))
    print(f"[Saved] Best -> {best_path} | Last -> {os.path.join(out_dir, 'last_epoch_resnet50.pth')}")

    # Save and plot history
    save_history(history, out_dir)
    plot_curves_pil(history, out_dir)

    return model


# =========================
# PIL-based Visual Helpers
# =========================
def _fit_into_box(img: Image.Image, w: int, h: int, bg=(245,245,245)) -> Image.Image:
    """Resize with aspect ratio and pad to exactly (w,h)."""
    im = img.copy()
    im.thumbnail((w, h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w, h), bg)
    x = (w - im.width) // 2
    y = (h - im.height) // 2
    canvas.paste(im, (x, y))
    return canvas

def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert normalized CHW tensor to PIL Image."""
    t = denorm(t, IMAGENET_MEAN, IMAGENET_STD).clamp(0,1)
    arr = (t.permute(1,2,0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)

def _blue_scale(v: float) -> tuple:
    """
    Map 0..1 to a light-to-deep blue (rough 'Blues' colormap).
    Returns RGB tuple.
    """
    v = max(0.0, min(1.0, float(v)))
    c0 = np.array([0xEE, 0xF3, 0xFB], dtype=np.float32)
    c1 = np.array([0x08, 0x45, 0x97], dtype=np.float32)
    rgb = (c0 + v*(c1 - c0)).astype(np.uint8)
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]))

def draw_confusion_matrix_pil(cm: np.ndarray, class_names: List[str], out_path: str, title: str = "Confusion Matrix"):
    """
    Draw a confusion matrix using PIL (no matplotlib).
    """
    n = cm.shape[0]
    if n <= 8:
        cell = 64
    elif n <= 16:
        cell = 48
    else:
        cell = 32

    margin_left = max(140, int(max(len(c) for c in class_names) * 8))
    margin_top  = 80
    margin_right = 40
    margin_bottom = 40

    width = margin_left + n*cell + margin_right
    height = margin_top + n*cell + margin_bottom

    img = Image.new("RGB", (width, height), (255,255,255))
    draw = ImageDraw.Draw(img)
    font_title = try_load_font(size=18)
    font_label = try_load_font(size=14)
    font_tick  = try_load_font(size=12)

    draw.text((10, 10), title, fill=(0,0,0), font=font_title)

    vmax = cm.max() if cm.size > 0 else 1
    vmax = max(1, vmax)
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            frac = float(v) / float(vmax)
            color = _blue_scale(frac)
            x0 = margin_left + j*cell
            y0 = margin_top + i*cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(220,220,220))
            txt = str(int(v))
            tw, th = draw.textsize(txt, font=font_tick)
            draw.text((x0 + (cell - tw)/2, y0 + (cell - th)/2), txt, fill=(0,0,0), font=font_tick)

    draw.text((margin_left + (n*cell)//2 - 40, 40), "Predicted", fill=(0,0,0), font=font_label)
    y_label = "True"
    yl = Image.new("RGBA", (80, 24), (255,255,255,0))
    yd = ImageDraw.Draw(yl)
    yd.text((0, 0), y_label, fill=(0,0,0), font=font_label)
    yl = yl.rotate(90, expand=True)
    img.paste(yl, (10, margin_top + (n*cell)//2 - yl.height//2), yl)

    for j, name in enumerate(class_names):
        label_img = Image.new("RGBA", (cell*2, 20), (255,255,255,0))
        ld = ImageDraw.Draw(label_img)
        ld.text((0,0), name, fill=(0,0,0), font=try_load_font(12))
        label_img = label_img.rotate(45, expand=True)
        lx = margin_left + j*cell + cell//2 - label_img.width//2
        ly = margin_top - label_img.height - 5
        img.paste(label_img, (lx, ly), label_img)

    for i, name in enumerate(class_names):
        tw, th = draw.textsize(name, font=try_load_font(12))
        x = margin_left - 8 - tw
        y = margin_top + i*cell + (cell - th)//2
        draw.text((x, y), name, fill=(0,0,0), font=try_load_font(12))

    ensure_dir(os.path.dirname(out_path))
    img.save(out_path)
    print(f"[Saved] {out_path}")


# =========================
# Evaluate
# =========================
def evaluate_and_plot(model, loader, class_names, device, out_dir="outputs", split_name="val"):
    ensure_dir(out_dir)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(y.numpy().tolist())

    print("\n[Classification Report]")
    report_text = classification_report(y_true, y_pred, target_names=class_names)
    print(report_text)

    with open(os.path.join(out_dir, f"classification_report_{split_name}.txt"), "w") as f:
        f.write(report_text)

    cm = confusion_matrix(y_true, y_pred)
    out_path = os.path.join(out_dir, f"confusion_matrix_{split_name}.png")
    draw_confusion_matrix_pil(cm, class_names, out_path, title=f"Confusion Matrix ({split_name})")


# =========================
# Visualizations (PIL)
# =========================
def annotate_and_save(img: Image.Image, text: str) -> Image.Image:
    """Add a semi-opaque text box at top-left."""
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)
    font = try_load_font(size=14)
    tw, th = draw.textsize(text, font=font)
    # This will draw a solid box on RGB; if you need true semi-transparency, composite with an RGBA overlay.
    draw.rectangle([(0,0), (tw+16, th+16)], fill=(0,0,0))
    draw.text((8,8), text, fill=(255,255,255), font=font)
    return draw_img

def show_and_save_grid(model, loader, class_names, device, out_dir="outputs",
                       split_name="val", n_rows=3, n_cols=4, tile=256):
    """
    Create a grid of predictions for a few samples from `loader`, using PIL only.
    """
    ensure_dir(out_dir)
    model.eval()
    total = n_rows * n_cols
    tiles = []
    shown = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            labels = labels.numpy()

            for i in range(inputs.size(0)):
                if shown >= total:
                    break
                img_pil = _tensor_to_pil(inputs[i].cpu())
                true_cls = class_names[int(labels[i])]
                pred_cls = class_names[int(preds[i])]
                conf = float(probs[i][int(preds[i])])
                annotated = annotate_and_save(img_pil, f"Pred: {pred_cls} ({conf:.2f}) | True: {true_cls}")
                fitted = _fit_into_box(annotated, tile, tile)
                tiles.append(fitted)
                shown += 1
            if shown >= total:
                break

    if not tiles:
        print("[Grid] No samples to show.")
        return

    gap = 8
    grid_w = n_cols*tile + (n_cols+1)*gap
    grid_h = n_rows*tile + (n_rows+1)*gap
    canvas = Image.new("RGB", (grid_w, grid_h), (245,245,245))

    for idx, im in enumerate(tiles):
        r, c = divmod(idx, n_cols)
        x = gap + c*(tile + gap)
        y = gap + r*(tile + gap)
        canvas.paste(im, (x, y))

    save_path = os.path.join(out_dir, f"grid_predictions_{split_name}.png")
    canvas.save(save_path)
    print(f"[Saved] {save_path}")

def predict_and_save_on_paths(model, img_paths: List[str], class_names: List[str], device, out_dir="outputs", label="uploaded"):
    ensure_dir(out_dir)
    save_root = os.path.join(out_dir, f"preds_{label}")
    ensure_dir(save_root)
    infer_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    results = []
    for p in img_paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"[Skip] {p} ({e})")
            continue
        x = infer_tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            prob = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
            pred_idx = int(prob.argmax())
            pred_cls = class_names[pred_idx]
            conf = float(prob[pred_idx])

        annotated = annotate_and_save(img, f"Pred: {pred_cls} ({conf:.2f})")
        out_path = os.path.join(save_root, os.path.basename(p))
        annotated.save(out_path)
        results.append({"path": p, "pred": pred_cls, "conf": conf, "out_path": out_path})

    with open(os.path.join(save_root, "predictions.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Saved] Annotated images under {save_root}")
    return results

def grid_from_annotated(folder: str, out_png: str, n_cols=4, max_images=16):
    files = []
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
        files.extend(glob(os.path.join(folder, ext)))
    files = sorted(files)[:max_images]
    if not files:
        print(f"[Grid] No images in {folder}")
        return
    imgs = [Image.open(p).convert("RGB") for p in files]
    n = len(imgs)
    n_cols = min(n_cols, n)
    n_rows = (n + n_cols - 1) // n_cols
    w, h = imgs[0].size
    target_w = 256
    target_h = int(h * (256 / w))
    imgs = [im.resize((target_w, target_h), Image.Resampling.LANCZOS) for im in imgs]

    gap = 8
    grid_w = n_cols*target_w + (n_cols+1)*gap
    grid_h = n_rows*target_h + (n_rows+1)*gap
    canvas = Image.new("RGB", (grid_w, grid_h), (245,245,245))

    for i, im in enumerate(imgs):
        r, c = divmod(i, n_cols)
        x = gap + c*(target_w + gap)
        y = gap + r*(target_h + gap)
        canvas.paste(im, (x, y))
    canvas.save(out_png)
    print(f"[Saved] {out_png}")


# =========================
# Main (CLI training)
# =========================
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # If RAW_DIR already has train/val, use it; else split into SPLIT_DIR
    if is_already_split(RAW_DIR):
        data_dir = RAW_DIR
        print("[Info] Using existing train/val in data dir")
    else:
        ensure_dir(SPLIT_DIR)
        auto_split_from_single_folder(RAW_DIR, SPLIT_DIR, split_ratio=SPLIT_RATIO)
        data_dir = SPLIT_DIR

    # Data & Loaders
    image_datasets, loaders, sizes, idx_to_class = make_dataloaders(
        data_dir=data_dir,
        input_size=INPUT_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    ensure_dir(OUTPUTS)
    with open(os.path.join(OUTPUTS, "class_names.json"), "w") as f:
        json.dump(class_names, f)
    print("[Saved] class_names.json", class_names)

    # Model (load if exists, else train) — fully offline
    best_path = os.path.join(OUTPUTS, "best_resnet50.pth")
    if os.path.exists(best_path):
        print(f"[Load] Found weights at {best_path}.")
        model = create_resnet50(num_classes=len(class_names),
                                freeze_backbone=False,
                                local_ckpt=None)
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.to(device).eval()
    else:
        model = create_resnet50(num_classes=len(class_names),
                                freeze_backbone=bool(LOCAL_RESNET50),
                                local_ckpt=LOCAL_RESNET50)
        model = train(model, loaders, sizes, device,
                      epochs=EPOCHS, lr=LR, unfreeze_after=UNFREEZE_AFTER, out_dir=OUTPUTS)

    # Evaluate & visualize on validation set
    evaluate_and_plot(model, loaders["val"], class_names, device, out_dir=OUTPUTS, split_name="val")
    show_and_save_grid(model, loaders["val"], class_names, device, out_dir=OUTPUTS,
                       split_name="val", n_rows=3, n_cols=4)

    # Predict on uploaded images (if any)
    uploads_dir = "./uploads"
    if os.path.isdir(uploads_dir):
        img_paths = []
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff"):
            img_paths.extend(glob(os.path.join(uploads_dir, ext)))
        if img_paths:
            print(f"[Uploads] Found {len(img_paths)} images in {uploads_dir}. Predicting...")
            _ = predict_and_save_on_paths(model, img_paths, class_names, device, out_dir=OUTPUTS, label="uploaded")
            annotated_folder = os.path.join(OUTPUTS, "preds_uploaded")
            grid_from_annotated(annotated_folder, os.path.join(OUTPUTS, "grid_predictions_uploaded.png"))
        else:
            print(f"[Uploads] No images found in {uploads_dir}.")
    else:
        print("[Uploads] Folder './uploads' not found. Create it and add images to auto-predict.")


# =========================
# STREAMLIT FRONTEND (UI)
# =========================
def run_streamlit_ui():
    """
    Launch Streamlit UI (CPU-only inference).
    Run with:
        streamlit run landcover_train_and_show.py -- --ui
    """
    import io
    import streamlit as st

    st.set_page_config(page_title="Landcover Classifier (Offline, CPU)", layout="wide")
    st.title("🏞️ Landcover Classifier — Offline • CPU")

    # Sidebar: model paths
    with st.sidebar:
        st.header("Model Settings")
        default_classes = "./outputs/class_names.json"
        default_weights = "./outputs/best_resnet50.pth"
        class_json = st.text_input("Class names JSON", default_classes)
        weights_path = st.text_input("Weights (.pth)", default_weights)

        model = None
        class_names = []
        err_msgs = []

        if not os.path.isfile(class_json):
            err_msgs.append("class_names.json not found.")
        if not os.path.isfile(weights_path):
            err_msgs.append("best_resnet50.pth not found.")

        if err_msgs:
            st.warning(" | ".join(err_msgs))
        else:
            try:
                class_names = json.load(open(class_json, "r"))
                st.success(f"Loaded classes: {class_names}")
            except Exception as e:
                st.error(f"Error reading class_names.json: {e}")

            try:
                # Always load on CPU to be safe
                tmp_model = create_resnet50(num_classes=len(class_names), freeze_backbone=False, local_ckpt=None)
                state = torch.load(weights_path, map_location="cpu")
                tmp_model.load_state_dict(state, strict=True)
                tmp_model.eval()
                model = tmp_model  # keep on CPU
                st.success("Model loaded successfully.")
            except Exception as e:
                st.error(f"Failed to load weights: {e}")
                model = None

    # Tabs
    tab1, tab2 = st.tabs(["🔼 Upload & Predict", "📊 Evaluate on Validation Set"])

    # --- Tab 1: Upload & Predict ---
    with tab1:
        st.subheader("Upload Images and Get Predictions")
        st.caption("Tip: You can select multiple images.")
        uploaded_files = st.file_uploader("Choose image(s)", type=["jpg","jpeg","png","bmp","tif","tiff"], accept_multiple_files=True)

        if uploaded_files and model is not None and class_names:
            infer_tfm = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

            cols = st.columns(3)
            for idx, uf in enumerate(uploaded_files):
                try:
                    img = Image.open(uf).convert("RGB")
                except Exception as e:
                    st.error(f"Could not open {uf.name}: {e}")
                    continue

                x = infer_tfm(img).unsqueeze(0)  # CPU tensor
                t0 = time.time()
                with torch.no_grad():
                    logits = model(x)
                    prob = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
                pred_idx = int(prob.argmax())
                pred_cls = class_names[pred_idx]
                conf = float(prob[pred_idx])
                dt_ms = (time.time() - t0) * 1000

                # Annotate preview
                vis = annotate_and_save(img, f"{pred_cls} ({conf:.2f})")
                cols[idx % 3].image(
                    vis,
                    caption=f"**Pred:** {pred_cls} ({conf:.2f}) • **Time:** {dt_ms:.1f} ms",
                    use_column_width=True
                )
        elif uploaded_files and (model is None or not class_names):
            st.info("Load valid class names and weights from the sidebar to run predictions.")

    # --- Tab 2: Evaluate on Validation Set ---
    with tab2:
        st.subheader("Evaluate on Validation Set")
        st.write("Point to your **validation folder** (e.g., `./data/data_split/val` or `./data/val`).")
        val_dir = st.text_input("Validation folder path", "./data/data_split/val")
        run_eval = st.button("Run Evaluation")

        if run_eval:
            if not model or not class_names:
                st.error("Load model and class names in the sidebar first.")
            elif not os.path.isdir(val_dir):
                st.error("Validation folder does not exist.")
            else:
                tf_eval = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(INPUT_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ])
                ds = datasets.ImageFolder(val_dir, transform=tf_eval)
                if len(ds) == 0:
                    st.warning("No images found in the validation folder.")
                else:
                    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)
                    y_true, y_pred = [], []
                    with torch.no_grad():
                        for x, y in loader:
                            logits = model(x)  # CPU
                            preds = logits.argmax(1).cpu().numpy()
                            y_pred.extend(preds.tolist())
                            y_true.extend(y.numpy().tolist())

                    acc = accuracy_score(y_true, y_pred)
                    report = classification_report(y_true, y_pred, target_names=ds.classes)
                    st.markdown(f"### ✅ Accuracy: **{acc*100:.2f}%**")
                    st.text_area("Classification Report", report, height=280)

                    # Confusion matrix (create PIL with existing helper, save to buffer, show)
                    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(ds.classes))))
                    tmp_png = os.path.join(OUTPUTS, "confusion_matrix_ui.png")
                    draw_confusion_matrix_pil(cm, ds.classes, tmp_png, title="Confusion Matrix (UI)")
                    # Read saved image into memory to show
                    try:
                        with open(tmp_png, "rb") as f:
                            st.image(f.read(), caption="Confusion Matrix", use_column_width=True)
                        # also offer download
                        st.download_button("Download Confusion Matrix", data=open(tmp_png, "rb").read(),
                                           file_name="confusion_matrix.png", mime="image/png")
                    except Exception:
                        # Fallback: load to PIL then display
                        st.image(Image.open(tmp_png), caption="Confusion Matrix", use_column_width=True)

    st.caption("Train first to produce `./outputs/best_resnet50.pth` and `class_names.json`, then use this UI.")


# =========================
# Entrypoint: choose CLI training vs UI
# =========================
if __name__ == "__main__":
    import sys
    # If called via: streamlit run landcover_train_and_show.py -- --ui
    if "--ui" in sys.argv:
        run_streamlit_ui()
    else:
        main()
