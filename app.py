from pathlib import Path

from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from inspection_log import (
    initialize_log,
    save_inspection,
    get_inspections,
)

from steelguard import (
    load_bundle,
    predict_image,
    add_few_shot_class,
)

st.set_page_config(page_title="SteelGuard AI", page_icon="🛡️", layout="wide")

# Initialize the local SQLite inspection database.
initialize_log()

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
    st.warning(
        "Model bundle not found. First prepare NEU-CLS and run `python train_model.py`, "
        "then restart the app."
    )
    st.code(
        "python train_model.py --data_dir data/NEU-CLS --held_out scratches\n"
        "streamlit run app.py"
    )
    st.stop()

model = bundle["model"]
known_classes = bundle["known_classes"]
held_out_class = bundle["held_out_class"]
threshold = float(bundle["unknown_threshold"])
device = bundle["device"]

# Load a saved few-shot prototype, if one exists.
new_proto = None
new_class_name = None
new_threshold = threshold
if NEW_CLASS_PATH.exists():
    try:
        new_data = torch.load(
            NEW_CLASS_PATH, map_location="cpu", weights_only=False
        )
        new_proto = new_data["prototype"].numpy()
        new_class_name = new_data["class_name"]
        new_threshold = float(new_data.get("threshold", threshold))
    except Exception as exc:
        st.warning(f"Could not load the saved learned-class prototype: {exc}")

st.sidebar.header("Inspection settings")
st.sidebar.metric("Known classes", len(known_classes))
st.sidebar.metric("Unknown threshold", f"{threshold:.3f}")
st.sidebar.write(f"Held-out class in benchmark: **{held_out_class}**")
st.sidebar.caption(
    "For a real deployment, threshold calibration should use validation data "
    "representative of production."
)

tab_inspect, tab_learn, tab_eval, tab_log, tab_about = st.tabs(
    [
        "🔎 Inspect image",
        "🧑‍🏭 Learn new defect",
        "📊 Evaluation",
        "🗂️ Inspection Log",
        "ℹ️ About",
    ]
)

with tab_inspect:
    st.subheader("Upload a steel surface image")

    # Select the production shift for this inspection.
    shift = st.selectbox(
        "Production Shift",
        ["Morning", "Evening", "Night"],
        key="inspection_shift",
    )

    uploaded = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        key="inspection_image_uploader",
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        c1, c2 = st.columns([1, 1.3])

        with c1:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with c2:
            if st.button("Inspect defect", type="primary"):
                with st.spinner(
                    "Extracting features and checking for unknown defects..."
                ):
                    result = predict_image(
                        model,
                        image,
                        known_classes,
                        bundle["prototypes"],
                        threshold,
                        device,
                        new_proto,
                        new_class_name,
                        new_threshold,
                    )

                if result["status"] == "known":
                    st.success(f"Known defect: {result['class_name']}")
                elif result["status"] == "learned":
                    st.success(
                        f"Learned defect recognized: {result['class_name']}"
                    )
                else:
                    st.error("UNKNOWN DEFECT — operator review required")

                st.metric(
                    "Nearest prototype distance", f"{result['distance']:.4f}"
                )
                st.metric(
                    "Classifier confidence", f"{result['confidence']:.1%}"
                )
                st.write(f"Decision: **{result['status'].upper()}**")
                st.write("Nearest known class:", result["nearest_known"])
                st.caption(
                    "Confidence is the closed-set classifier probability; it is "
                    "not the probability that the image is safe or correctly identified."
                )

                # Keep the most recent inspection available in the session.
                st.session_state["last_result"] = result
                st.session_state["last_image"] = image

                # Save this inspection only when the operator clicks Inspect defect.
                logged_class = (
                    result.get("class_name")
                    if result["status"] in ("known", "learned")
                    else "unknown"
                )
                save_inspection(
                    image_name=uploaded.name,
                    predicted_class=str(logged_class),
                    confidence=float(result["confidence"]),
                    shift=shift,
                    status=str(result["status"]),
                )
                st.success(f"Inspection saved to the log for the {shift} shift.")

with tab_learn:
    st.subheader("Human-in-the-loop few-shot update")
    st.write(
        "Upload five labelled examples of one new defect class. The app averages "
        "their embeddings to create a prototype; the neural network is not fully retrained."
    )
    class_name = st.text_input("New defect class name", value=held_out_class)
    few_shot_files = st.file_uploader(
        "Upload exactly five labelled example images",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        key="fewshot",
    )

    if few_shot_files:
        st.info(
            f"{len(few_shot_files)} image(s) selected. Exactly five are required."
        )
        cols = st.columns(min(5, len(few_shot_files)))
        for i, f in enumerate(few_shot_files[:5]):
            with cols[i % len(cols)]:
                st.image(
                    Image.open(f),
                    caption=f"Example {i + 1}",
                    use_container_width=True,
                )

    if st.button("Learn new defect from five examples", type="primary"):
        if not class_name.strip():
            st.error("Enter a class name.")
        elif not few_shot_files or len(few_shot_files) != 5:
            st.error("Upload exactly five examples.")
        elif class_name.strip() in known_classes:
            st.error(
                "This class is already in the known class list. Choose a new class name."
            )
        else:
            images = [Image.open(f).convert("RGB") for f in few_shot_files]
            prototype, learned_threshold = add_few_shot_class(
                model, images, device, base_threshold=threshold
            )

            MODEL_DIR.mkdir(exist_ok=True)
            torch.save(
                {
                    "prototype": torch.tensor(prototype),
                    "class_name": class_name.strip(),
                    "threshold": learned_threshold,
                },
                NEW_CLASS_PATH,
            )

            st.success(
                f"Added prototype for '{class_name.strip()}'. "
                f"Learned threshold: {learned_threshold:.3f}"
            )
            st.warning(
                "Test on separate images that were not used for learning."
            )

    if new_proto is not None:
        st.info(f"Currently learned class: **{new_class_name}**")
        if st.button("Remove learned class prototype"):
            NEW_CLASS_PATH.unlink(missing_ok=True)
            st.success("Prototype removed. Rerun the app to refresh.")
            st.rerun()

with tab_eval:
    st.subheader("Benchmark evaluation")
    st.write(
        "This benchmark uses the reserved held-out class as unknown. The test set "
        "must not include the five operator examples."
    )
    eval_dir = st.text_input(
        "Path to evaluation image folder", value="data/NEU-CLS/test"
    )
    st.caption(
        "Expected folder structure: one subfolder per true class, containing images. "
        "The benchmark evaluates the original five-class open-set model, so a saved "
        "few-shot prototype is deliberately not applied here. Test learned-class "
        "generalisation separately in the Inspect image tab."
    )

    if st.button("Run evaluation"):
        folder = Path(eval_dir)

        if not folder.exists():
            st.error("Folder not found. Update the path to your dataset test directory.")
        else:
            rows = []
            y_true = []
            y_pred = []
            valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

            # Explicit label order guarantees that "unknown" appears in the matrix,
            # even if no test image is predicted as unknown.
            matrix_labels = list(known_classes) + ["unknown"]

            for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
                for fp in sorted(sub.iterdir()):
                    if not fp.is_file() or fp.suffix.lower() not in valid_ext:
                        continue

                    try:
                        img = Image.open(fp).convert("RGB")

                        # Evaluate the original model without the learned class.
                        res = predict_image(
                            model,
                            img,
                            known_classes,
                            bundle["prototypes"],
                            threshold,
                            device,
                            None,
                            None,
                            threshold,
                        )

                        true_folder_label = sub.name
                        actual_label = (
                            true_folder_label
                            if true_folder_label in known_classes
                            else "unknown"
                        )
                        predicted_label = (
                            res["class_name"]
                            if res["status"] == "known"
                            else "unknown"
                        )

                        if predicted_label not in matrix_labels:
                            predicted_label = "unknown"

                        y_true.append(actual_label)
                        y_pred.append(predicted_label)

                        rows.append(
                            {
                                "file": fp.name,
                                "actual": true_folder_label,
                                "actual_for_matrix": actual_label,
                                "predicted": predicted_label,
                                "distance": round(float(res["distance"]), 4),
                                "status": res["status"],
                            }
                        )

                    except Exception as exc:
                        rows.append(
                            {
                                "file": fp.name,
                                "actual": sub.name,
                                "actual_for_matrix": None,
                                "predicted": "ERROR",
                                "distance": None,
                                "status": str(exc),
                            }
                        )

            successful_count = len(y_true)

            if successful_count:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)

                cm = confusion_matrix(
                    y_true,
                    y_pred,
                    labels=matrix_labels,
                )

                st.subheader("Confusion matrix — known classes and unknown")
                fig, ax = plt.subplots(figsize=(10, 7))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    xticklabels=matrix_labels,
                    yticklabels=matrix_labels,
                    ax=ax,
                    cmap="Blues",
                )
                ax.set_xlabel("Predicted label")
                ax.set_ylabel("Actual label")
                ax.set_title("Open-set defect detection confusion matrix")
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                plt.setp(ax.get_yticklabels(), rotation=0)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.download_button(
                    "Download evaluation CSV",
                    df.to_csv(index=False),
                    "steelguard_evaluation.csv",
                    "text/csv",
                )
                st.metric("Images evaluated successfully", successful_count)
                st.metric(
                    "Images with processing errors",
                    len(df) - successful_count,
                )

                accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))
                st.metric("Open-set accuracy", f"{accuracy:.1%}")

                report = classification_report(
                    y_true,
                    y_pred,
                    labels=matrix_labels,
                    output_dict=True,
                    zero_division=0,
                )
                report_df = pd.DataFrame(report).transpose()
                st.subheader("Per-class evaluation metrics")
                st.dataframe(report_df.round(3), use_container_width=True)

                st.caption(
                    "The 'unknown' row includes the held-out scratches class. "
                    "This benchmark intentionally disables the saved few-shot prototype "
                    "to measure the original model's unknown detection. To evaluate the "
                    "learned scratches class, use separate scratches images that were "
                    "not included among the five operator examples."
                )
            else:
                st.warning(
                    "No valid images were successfully evaluated. Check that the folder "
                    "contains class subfolders with supported image files."
                )

with tab_log:
    st.subheader("Shift-wise Inspection History")
    st.write(
        "Review inspections saved when an operator clicks **Inspect defect**. "
        "Records are stored in the local SQLite database."
    )

    selected_shift = st.selectbox(
        "Filter inspection history by shift",
        ["All", "Morning", "Evening", "Night"],
        key="log_shift_filter",
    )
    log_df = get_inspections(selected_shift)

    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Inspections in this view", len(log_df))
    if not log_df.empty:
        metric_col2.metric(
            "Different predicted classes",
            int(log_df["predicted_class"].nunique()),
        )
        st.dataframe(log_df, use_container_width=True)

        csv_data = log_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download inspection history (CSV)",
            data=csv_data,
            file_name="steelguard_inspection_log.csv",
            mime="text/csv",
        )

        counts = (
            log_df.groupby("predicted_class")
            .size()
            .sort_values(ascending=False)
            .rename("inspection_count")
            .reset_index()
        )
        st.subheader("Defect counts in selected shift")
        st.bar_chart(counts.set_index("predicted_class"))
    else:
        metric_col2.metric("Different predicted classes", 0)
        st.info("No inspections recorded for this shift yet.")

with tab_about:
    st.subheader("System design")
    st.markdown(
        """
        - **Backbone:** ImageNet-pretrained ResNet-18, used as a feature extractor.
        - **Known-class model:** Logistic regression on embeddings from five classes.
        - **Unknown detection:** cosine distance to the nearest known-class prototype and a validation-calibrated threshold.
        - **Few-shot update:** mean normalized embedding of five operator-labelled images.
        - **Safety behavior:** unknown predictions are routed to operator review rather than assigned a known class.

        This is a hackathon prototype, not a certified industrial inspection system. Performance must be validated on representative production data before operational use.
        """
    )
    st.write("Known classes:", ", ".join(known_classes))
    st.write("Held-out class:", held_out_class)

