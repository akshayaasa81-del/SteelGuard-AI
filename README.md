# SteelGuard AI — NEXAI 2026 PS 2.3

A hackathon prototype for unknown-defect detection and five-example prototype learning using NEU-CLS steel surface defect images.

## Requirements
- Python 3.10 or 3.11 recommended
- 8 GB RAM recommended
- GPU optional; CPU works but embedding extraction may be slower
- Internet may be needed on the first run to download pretrained ResNet-18 weights

## 1. Create environment

### Windows PowerShell
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you already use another Python version, check PyTorch compatibility before installing.

## 2. Download and arrange NEU-CLS

Use the NEU Surface Defect Database / NEU-CLS source referenced by the NEXAI problem statement. Extract the classification dataset so the folders look like:

```text
data/
  NEU-CLS/
    crazing/
      image1.jpg
    inclusion/
    patches/
    pitted_surface/
    rolled-in_scale/
    scratches/
```

Folder names become class labels. If your downloaded archive uses different folder names, use the exact folder name in `--held_out`.

## 3. Train the baseline and calibrate unknown detection

```bash
python train_model.py --data_dir data/NEU-CLS --held_out scratches
```

The script:
- Extracts normalized embeddings with ImageNet-pretrained ResNet-18.
- Holds out the selected class from classifier training.
- Trains a logistic-regression classifier on the five remaining classes.
- Calculates class prototypes.
- Sets a distance threshold from known-class validation embeddings.
- Reports known-class validation accuracy, held-out unknown detection rate and AUROC.
- Saves the model bundle under `artifacts/`.

The first run can take several minutes and may download ResNet-18 weights. For repeat runs, the downloaded weights are normally cached.

## 4. Run the app

```bash
streamlit run app.py
```

## 5. Demonstrate the required workflow

1. Upload a known-class image and inspect its predicted class.
2. Upload a held-out-class image; it should be treated as unknown when its nearest-prototype distance exceeds the calibrated threshold.
3. In **Learn new defect**, enter the class name and upload exactly five labelled examples.
4. The app averages their normalized embeddings into a new prototype.
5. Upload separate images from that class and check whether the new prototype recognizes them.
6. Use **Evaluation** to evaluate folders of labelled images.

## Dataset split and evaluation caveats

For the hackathon's final reported numbers, keep training, threshold-calibration, five operator examples and final test images separate. The starter script uses a stratified split of the five known classes and the held-out class as the unknown pool. The printed held-out AUROC is a prototype diagnostic; for a final benchmark, establish a separate held-out test set and freeze the threshold before evaluating it.

The provided app's evaluation tab is a basic folder-based inspection report. Adapt its folder path and split names to your exact local dataset layout. For a rigorous final confusion matrix, label held-out examples as `unknown` before the five-shot update and separately report learned-class performance after the update.

## Limitations
- NEU-CLS is a small grayscale research dataset, not a guarantee of real factory performance.
- A distance threshold can reject known images or accept unknown ones; report both errors.
- A prototype update is lightweight few-shot adaptation, not full neural-network retraining.
- This is not a certified safety or production inspection system.

## Suggested team split
- ML baseline and embeddings
- Unknown detection and few-shot update
- Streamlit UI and integration
- Dataset checks, evaluation, documentation and pitch
