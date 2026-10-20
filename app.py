import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from steelguard import (
    load_bundle, predict_image, add_few_shot_class, evaluate_unknown_detection,
    prepare_image, get_embedding
)

st.set_page_config(page_title="SteelGuard AI", page_icon="🛡️", layout="wide")
st.title("🛡️ SteelGuard AI")
st.caption("Open-world steel surface defect inspection • NEXAI 2026 PS 2.3")

MODEL_DIR = Path("artifacts")
BUNDLE_PATH = MODEL_DIR / "steelguard_bundle.pt"
LABELS_PATH = MODEL_DIR / "class_labels.json"
NEW_CLASS_PATH = MODEL_DIR / "new_class_prototype.pt"

@st.cache_resource
def get_bundle():
    if not BUNDLE_PATH.exists():
        return None
    return load_bundle(BUNDLE_PATH)

bundle = get_bundle()
if bundle is None:
    st.warning("Model bundle not found. First prepare NEU-CLS and run `python train_model.py`, then restart the app.")
    st.code("python train_model.py --data_dir data/NEU-CLS --held_out scratches\nstreamlit run app.py")
    st.stop()

model = bundle["model"]
known_classes = bundle["known_classes"]
held_out_class = bundle["held_out_class"]
threshold = float(bundle["unknown_threshold"])
device = bundle["device"]

if NEW_CLASS_PATH.exists():
    try:
        new_data = torch.load(NEW_CLASS_PATH, map_location="cpu", weights_only=False)
        new_proto = new_data["prototype"].numpy()
        new_class_name = new_data["class_name"]
    except Exception:
        new_proto, new_class_name = None, None
else:
    new_proto, new_class_name = None, None

st.sidebar.header("Inspection settings")
st.sidebar.metric("Known classes", len(known_classes))
st.sidebar.metric("Unknown threshold", f"{threshold:.3f}")
st.sidebar.write(f"Held-out class in benchmark: **{held_out_class}**")
st.sidebar.caption("For a real deployment, threshold calibration should use validation data representative of production.")

tab_inspect, tab_learn, tab_eval, tab_about = st.tabs(
    ["🔎 Inspect image", "🧑‍🏭 Learn new defect", "📊 Evaluation", "ℹ️ About"]
)

with tab_inspect:
    st.subheader("Upload a steel surface image")
    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        c1, c2 = st.columns([1, 1.3])
        with c1:
            st.image(image, caption="Uploaded image", use_container_width=True)
        with c2:
            if st.button("Inspect defect", type="primary"):
                with st.spinner("Extracting features and checking for unknown defects..."):
                    result = predict_image(model, image, known_classes, bundle["prototypes"],
                                           threshold, device, new_proto, new_class_name)
                if result["status"] == "known":
                    st.success(f"Known defect: {result['class_name']}")
                elif result["status"] == "learned":
                    st.success(f"Learned defect recognized: {result['class_name']}")
                else:
                    st.error("UNKNOWN DEFECT — operator review required")
                st.metric("Nearest prototype distance", f"{result['distance']:.4f}")
                st.metric("Classifier confidence", f"{result['confidence']:.1%}")
                st.write(f"Decision: **{result['status'].upper()}**")
                st.write("Nearest known class:", result["nearest_known"])
                st.caption("Confidence is the closed-set classifier probability; it is not the probability that the image is safe or correctly identified.")
                st.session_state["last_result"] = result
                st.session_state["last_image"] = image

with tab_learn:
    st.subheader("Human-in-the-loop few-shot update")
    st.write("Upload five labelled examples of one new defect class. The app averages their embeddings to create a prototype; the neural network is not fully retrained.")
    class_name = st.text_input("New defect class name", value=held_out_class)
    few_shot_files = st.file_uploader(
        "Upload exactly five labelled example images",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        key="fewshot"
    )
    if few_shot_files:
        st.info(f"{len(few_shot_files)} image(s) selected. Exactly five are required.")
        cols = st.columns(min(5, len(few_shot_files)))
        for i, f in enumerate(few_shot_files[:5]):
            with cols[i % len(cols)]:
                st.image(Image.open(f), caption=f"Example {i+1}", use_container_width=True)
    if st.button("Learn new defect from five examples", type="primary"):
        if not class_name.strip():
            st.error("Enter a class name.")
        elif not few_shot_files or len(few_shot_files) != 5:
            st.error("Upload exactly five examples.")
        elif class_name.strip() in known_classes:
            st.error("This class is already in the known class list. Choose a new class name.")
        else:
            images = [Image.open(f).convert("RGB") for f in few_shot_files]
            prototype = add_few_shot_class(model, images, device)
            MODEL_DIR.mkdir(exist_ok=True)
            torch.save({"prototype": torch.tensor(prototype), "class_name": class_name.strip()}, NEW_CLASS_PATH)
            st.cache_resource.clear()
            st.success(f"Added prototype for '{class_name.strip()}'. Upload separate test images in Inspect image to verify generalization.")
            st.warning("For a fair evaluation, do not reuse these five images as test images.")

    if new_proto is not None:
        st.info(f"Currently learned class: **{new_class_name}**")
        if st.button("Remove learned class prototype"):
            NEW_CLASS_PATH.unlink(missing_ok=True)
            st.success("Prototype removed. Rerun the app to refresh.")
            st.rerun()

with tab_eval:
    st.subheader("Benchmark evaluation")
    st.write("This benchmark uses the reserved held-out class as unknown. The test set must not include the five operator examples.")
    eval_dir = st.text_input("Path to evaluation image folder", value="data/NEU-CLS/test")
    st.caption("Expected folder structure: one subfolder per true class, containing images. Only classes present in this directory will be evaluated.")
    if st.button("Run evaluation"):
        folder = Path(eval_dir)
        if not folder.exists():
            st.error("Folder not found. Update the path to your dataset test directory.")
        else:
            rows = []
            y_true, y_pred, unknown_scores, binary_true = [], [], [], []
            valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
                for fp in sorted(sub.iterdir()):
                    if fp.suffix.lower() not in valid_ext:
                        continue
                    try:
                        img = Image.open(fp).convert("RGB")
                        res = predict_image(model, img, known_classes, bundle["prototypes"],
                                            threshold, device, new_proto, new_class_name)
                        true = sub.name
                        is_unknown = true == held_out_class
                        predicted = res["class_name"] if res["status"] in ("known", "learned") else "unknown"
                        if is_unknown and res["status"] != "learned":
                            predicted = "unknown"
                        y_true.append(true if true in known_classes else "unknown")
                        y_pred.append(predicted if predicted in known_classes or predicted == "unknown" else "learned_new")
                        unknown_scores.append(res["distance"])
                        binary_true.append(1 if is_unknown else 0)
                        rows.append({"file": fp.name, "actual": true, "predicted": predicted,
                                     "distance": round(res["distance"], 4), "status": res["status"]})
                    except Exception as e:
                        rows.append({"file": fp.name, "actual": sub.name, "predicted": "ERROR", "distance": None, "status": str(e)})
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)
                labels = sorted(set(y_true) | set(y_pred))
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, ax=ax, cmap="Blues")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
                st.download_button("Download evaluation CSV", df.to_csv(index=False), "steelguard_evaluation.csv", "text/csv")
                st.metric("Images evaluated", len(df))
                st.caption("Interpret metrics with care: the threshold and class splits must be fixed before final testing. For the learned-class test, evaluate separately on held-out examples not used in the five-shot update.")
            else:
                st.warning("No images found in subfolders.")

with tab_about:
    st.subheader("System design")
    st.markdown("""
    - **Backbone:** ImageNet-pretrained ResNet-18, used as a feature extractor.
    - **Known-class model:** Logistic regression on embeddings from five classes.
    - **Unknown detection:** cosine distance to the nearest known-class prototype and a validation-calibrated threshold.
    - **Few-shot update:** mean normalized embedding of five operator-labelled images.
    - **Safety behavior:** unknown predictions are routed to operator review rather than assigned a known class.

    This is a hackathon prototype, not a certified industrial inspection system. Performance must be validated on representative production data before operational use.
    """)
    st.write("Known classes:", ", ".join(known_classes))
    st.write("Held-out class:", held_out_class)
