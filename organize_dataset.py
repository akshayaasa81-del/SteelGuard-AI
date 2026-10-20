
from pathlib import Path
import shutil

# Downloaded dataset location
SOURCE = Path.home() / "Downloads" / "NEU-CLS"

# Destination inside your project
DEST = Path("data") / "NEU-CLS"

classes = {
    "crazing": "crazing",
    "inclusion": "inclusion",
    "patches": "patches",
    "pitted_surface": "pitted_surface",
    "rolled-in_scale": "rolled-in_scale",
    "scratches": "scratches",
}

# Find images in both train and valid
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
copied = {name: 0 for name in classes}

for split in ("train", "valid"):
    split_dir = SOURCE / split

    if not split_dir.exists():
        print(f"Folder not found: {split_dir}")
        continue

    for image in split_dir.rglob("*"):
        if not image.is_file() or image.suffix.lower() not in extensions:
            continue

        filename = image.stem.lower()

        for class_name in classes:
            if filename.startswith(class_name + "_"):
                target_dir = DEST / classes[class_name]
                target_dir.mkdir(parents=True, exist_ok=True)

                target = target_dir / image.name

                # Avoid overwriting an image with the same name
                if target.exists():
                    target = target_dir / f"{split}_{image.name}"

                shutil.copy2(image, target)
                copied[class_name] += 1
                break

print("\nDataset organization complete:")
for name, count in copied.items():
    print(f"{name}: {count} images")

print(f"\nDestination: {DEST.resolve()}")

if sum(copied.values()) == 0:
    print("No images were copied. Check the source folder path.")
