import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import normalize
from steelguard import build_backbone, get_embedding, fit_prototypes, cosine_distance

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def collect_images(root):
    rows = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in VALID_EXT:
                rows.append((path, class_dir.name))
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Root folder containing one subfolder per defect class")
    parser.add_argument("--held_out", default="scratches", help="Class held out entirely from model training")
    parser.add_argument("--threshold_quantile", type=float, default=0.95)
    parser.add_argument("--output_dir", default="artifacts")
    args = parser.parse_args()

    root = Path(args.data_dir)
    if not root.exists():
        raise SystemExit(f"Dataset directory not found: {root}")
    rows = collect_images(root)
    if not rows:
        raise SystemExit("No images found. Expected data_dir/class_name/image.jpg")
    classes = sorted(set(label for _, label in rows))
    if len(classes) < 2:
        raise SystemExit("Expected at least two class folders.")
    if args.held_out not in classes:
        raise SystemExit(f"Held-out class '{args.held_out}' not found. Available classes: {classes}")

    known_classes = [c for c in classes if c != args.held_out]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Known classes: {known_classes}")
    print(f"Held out as unknown: {args.held_out}")
    print("Building feature extractor; first run may download pretrained weights.")

    backbone = build_backbone(device)
    cache = {}
    embeddings, labels, paths = [], [], []
    held_embeddings, held_paths = [], []
    for i, (path, label) in enumerate(rows, start=1):
        try:
            image = Image.open(path).convert("RGB")
            emb = get_embedding(backbone, image, device)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue
        cache[path] = emb
        if label == args.held_out:
            held_embeddings.append(emb)
            held_paths.append(path)
        else:
            embeddings.append(emb)
            labels.append(label)
            paths.append(path)
        if i % 100 == 0:
            print(f"Embedded {i}/{len(rows)} images")

    X = np.stack(embeddings)
    y = np.array(labels)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced")
    classifier.fit(X_train, y_train)
    val_pred = classifier.predict(X_val)
    print("\nKnown-class validation accuracy:", round(accuracy_score(y_val, val_pred), 4))
    print(classification_report(y_val, val_pred, zero_division=0))

    # Fit prototypes on training split only to avoid leaking validation examples.
    proto = fit_prototypes(X_train, y_train, known_classes)
    val_dist = np.array([min(cosine_distance(x, p) for p in proto.values()) for x in X_val])
    held = np.stack(held_embeddings)
    held_dist = np.array([min(cosine_distance(x, p) for p in proto.values()) for x in held])

    # Calibrate a conservative threshold on known validation distances.
    threshold = float(np.quantile(val_dist, args.threshold_quantile))
    binary_true = np.concatenate([np.zeros(len(val_dist)), np.ones(len(held_dist))])
    unknown_score = np.concatenate([val_dist, held_dist])
    try:
        auc = roc_auc_score(binary_true, unknown_score)
    except Exception:
        auc = float("nan")
    print(f"Unknown threshold (known validation q={args.threshold_quantile}): {threshold:.4f}")
    print(f"Known-vs-held-out AUROC (higher distance = more unknown): {auc:.4f}")
    print(f"Known validation accepted at threshold: {(val_dist <= threshold).mean():.1%}")
    print(f"Held-out class flagged unknown: {(held_dist > threshold).mean():.1%}")
    print("NOTE: AUROC uses held-out-class images as unknown and known validation images as known.")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundle = {
        "backbone": backbone.cpu(),
        "classifier": classifier,
        "known_classes": known_classes,
        "held_out_class": args.held_out,
        "prototypes": proto,
        "unknown_threshold": threshold,
        "threshold_quantile": args.threshold_quantile
    }
    torch.save(bundle, output / "steelguard_bundle.pt")
    (output / "class_labels.json").write_text(json.dumps({
        "known_classes": known_classes, "held_out_class": args.held_out,
        "unknown_threshold": threshold
    }, indent=2))
    print(f"\nSaved model bundle to {output / 'steelguard_bundle.pt'}")
    print("Run: streamlit run app.py")

if __name__ == "__main__":
    main()
