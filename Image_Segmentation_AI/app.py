"""
app.py
------
Main Streamlit application for:

    "AI-Based Image Segmentation Using Rule-Based and
     Traditional Image Processing Techniques"

Architecture implemented in this file (wiring together the other modules):

    Input Image
        |
    Preprocessing        (preprocessing.py)
        |
    Image Segmentation   (segmentation.py)
        |
    Region Detection     (feature_extraction.py - connected components)
        |
    Feature Extraction   (feature_extraction.py - geometric features)
        |
    Knowledge Base        (rule_engine.KnowledgeBase)
        |
    Rule-Based Reasoning  (rule_engine.run_rule_engine - THE AI COMPONENT)
        |
    Decision              (SELECTED / REJECTED per region)
        |
    Final Segmented Image (drawn with OpenCV)

IMPORTANT (for viva / academic clarity):
    OpenCV is used ONLY for classical image processing (grayscale
    conversion, thresholding, edge detection, morphology, connected
    components). It is NOT an AI model. The AI in this project is the
    knowledge-based rule engine in rule_engine.py, which reasons over
    region features using explicit IF-THEN rules (forward-chaining
    style). No machine learning or deep learning is used anywhere.
"""

import time
import io

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from preprocessing import preprocess_pipeline
from segmentation import segment_image
from feature_extraction import extract_regions
from rule_engine import KnowledgeBase, run_rule_engine, summarize_decisions
from evaluation import evaluate_segmentation


# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AI-Based Image Segmentation",
    layout="wide",
)

st.title("AI-Based Image Segmentation")
st.caption("Traditional AI + Classical Image Processing")

st.markdown(
    "This project performs image segmentation using **classical image "
    "processing** (OpenCV) and applies **traditional AI reasoning** "
    "(a knowledge-based rule engine) to decide which segmented regions "
    "satisfy defined IF-THEN rules. No machine learning or deep learning "
    "is used anywhere in this pipeline."
)


# ----------------------------------------------------------------------
# SIDEBAR - SETTINGS (Knowledge Base + Segmentation controls)
# ----------------------------------------------------------------------
st.sidebar.header("Settings")

segmentation_method = st.sidebar.selectbox(
    "Select Segmentation Method",
    [
        "Binary Thresholding",
        "Otsu Thresholding",
        "Adaptive Thresholding",
        "Edge-Based Segmentation",
        "Region-Based Segmentation",
    ],
)

enable_preprocessing = st.sidebar.checkbox("Enable Preprocessing (Gaussian Blur)", value=True)
blur_kernel = st.sidebar.slider("Preprocessing Blur Kernel Size", 1, 15, 5, step=2)

st.sidebar.subheader("Segmentation Parameters")

threshold_value = st.sidebar.slider("Threshold Value (Binary / Region-Based)", 0, 255, 127)
block_size = st.sidebar.slider("Adaptive Block Size (odd)", 3, 51, 11, step=2)
c_value = st.sidebar.slider("Adaptive Constant C", -20, 20, 2)
low_thresh = st.sidebar.slider("Canny Low Threshold", 0, 255, 50)
high_thresh = st.sidebar.slider("Canny High Threshold", 0, 255, 150)
kernel_size = st.sidebar.slider("Edge Closing Kernel Size", 1, 21, 3, step=2)

st.sidebar.subheader("Knowledge Base (AI Rule Engine)")
minimum_area = st.sidebar.number_input("Minimum Region Area", min_value=0, value=500, step=10)
minimum_width = st.sidebar.number_input("Minimum Width", min_value=0, value=20, step=1)
minimum_height = st.sidebar.number_input("Minimum Height", min_value=0, value=20, step=1)

st.sidebar.subheader("Evaluation (Optional)")
gt_file = st.sidebar.file_uploader(
    "Upload Ground-Truth Mask (optional)", type=["jpg", "jpeg", "png"], key="gt_mask"
)

segment_button = st.sidebar.button("SEGMENT IMAGE", type="primary")


# ----------------------------------------------------------------------
# MODULE 1 - IMAGE INPUT
# ----------------------------------------------------------------------
st.header("1. Upload Image")

uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

use_sample = st.checkbox("Use built-in sample image instead", value=not bool(uploaded_file))


def load_image_from_upload(file) -> np.ndarray:
    """Decode an uploaded Streamlit file into an OpenCV BGR image."""
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


image_bgr = None
file_name = None

try:
    if uploaded_file is not None and not use_sample:
        image_bgr = load_image_from_upload(uploaded_file)
        file_name = uploaded_file.name
        if image_bgr is None:
            st.error("The uploaded file could not be read as a valid image. Please try a different file.")
    else:
        image_bgr = cv2.imread("input_images/sample.jpg")
        file_name = "sample.jpg (built-in)"
        if image_bgr is None:
            st.error(
                "Built-in sample image not found. Please run 'python generate_sample.py' "
                "or upload your own image."
            )
except Exception as e:
    st.error(f"Error while loading the image: {e}")
    image_bgr = None


# ----------------------------------------------------------------------
# DISPLAY ORIGINAL IMAGE INFO
# ----------------------------------------------------------------------
if image_bgr is not None:
    if image_bgr.size == 0:
        st.error("The uploaded image appears to be empty. Please upload a valid image.")
        st.stop()

    h, w = image_bgr.shape[:2]
    channels = 1 if len(image_bgr.shape) == 2 else image_bgr.shape[2]

    if h < 10 or w < 10:
        st.error("The uploaded image is too small to process. Please upload a larger image.")
        st.stop()

    st.header("2. Original Image")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
    with col2:
        st.markdown("**Image Information**")
        st.write(f"File Name: `{file_name}`")
        st.write(f"Width: `{w}` px")
        st.write(f"Height: `{h}` px")
        st.write(f"Channels: `{channels}`")
else:
    st.info("Please upload an image or enable the built-in sample image to continue.")
    st.stop()


# ----------------------------------------------------------------------
# MODULE 2 - PREPROCESSING
# ----------------------------------------------------------------------
st.header("3. Preprocessing")

try:
    pre = preprocess_pipeline(image_bgr, enable_preprocessing=enable_preprocessing, blur_kernel=blur_kernel)
except Exception as e:
    st.error(f"Preprocessing failed: {e}")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    st.image(pre["rgb"], caption="Original Image (RGB)", use_container_width=True)
with col2:
    st.image(pre["preprocessed"], caption="Preprocessed Image (Grayscale + Blur)" if enable_preprocessing
              else "Preprocessed Image (Grayscale only)", use_container_width=True, clamp=True)


# ----------------------------------------------------------------------
# RUN FULL PIPELINE (triggered by button, but also runs once by default
# with default params so the page isn't empty on first load)
# ----------------------------------------------------------------------
st.header("4. Segmentation Result")

params = {
    "threshold_value": threshold_value,
    "block_size": block_size,
    "c": c_value,
    "low_thresh": low_thresh,
    "high_thresh": high_thresh,
    "kernel_size": kernel_size,
}

start_time = time.time()

try:
    final_mask = segment_image(pre["preprocessed"], segmentation_method, params)
except Exception as e:
    st.error(f"Segmentation failed: {e}")
    st.stop()

st.image(final_mask, caption=f"Segmentation Mask ({segmentation_method})", use_container_width=True, clamp=True)


# ----------------------------------------------------------------------
# MODULE 5 - CONNECTED COMPONENT ANALYSIS / REGION DETECTION
# ----------------------------------------------------------------------
st.header("5. Detected Regions")

try:
    regions_df, labels, num_labels = extract_regions(final_mask)
except Exception as e:
    st.error(f"Region detection failed: {e}")
    st.stop()

if len(regions_df) == 0:
    st.warning(
        "No significant regions were detected. Try reducing the threshold, "
        "changing the segmentation method, or adjusting the minimum area/width/height."
    )
else:
    st.markdown(f"**Total Regions Detected (excluding background): {len(regions_df)}**")
    display_df = regions_df.rename(columns={
        "region_id": "Region ID",
        "area": "Area",
        "width": "Width",
        "height": "Height",
        "aspect_ratio": "Aspect Ratio",
        "centroid_x": "Centroid X",
        "centroid_y": "Centroid Y",
        "x": "BBox X",
        "y": "BBox Y",
    })
    st.dataframe(display_df, use_container_width=True)

    # Visualize all raw detected regions with bounding boxes (before rules)
    regions_vis = pre["rgb"].copy()
    for _, r in regions_df.iterrows():
        x, y, rw, rh = int(r["x"]), int(r["y"]), int(r["width"]), int(r["height"])
        cv2.rectangle(regions_vis, (x, y), (x + rw, y + rh), (255, 165, 0), 2)
        cv2.putText(regions_vis, str(int(r["region_id"])), (x, max(0, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
    st.image(regions_vis, caption="All Detected Regions (before AI rule filtering)", use_container_width=True)


# ----------------------------------------------------------------------
# MODULE 6-8 - TRADITIONAL AI: KNOWLEDGE BASE + RULE ENGINE + REASONING
# ----------------------------------------------------------------------
st.header("6. AI Rule-Based Decision (Traditional AI Reasoning)")

st.markdown(
    "**Knowledge Base:** `minimum_area = {}`, `minimum_width = {}`, "
    "`minimum_height = {}`".format(minimum_area, minimum_width, minimum_height)
)

kb = KnowledgeBase(
    minimum_area=minimum_area,
    minimum_width=minimum_width,
    minimum_height=minimum_height,
)

decisions = run_rule_engine(regions_df, kb)

if len(decisions) == 0:
    st.info("No regions available for the rule engine to reason over.")
else:
    for d in decisions:
        with st.expander(f"Region {int(d['region_id'])} — Final Decision: {d['decision']}"):
            for step in d["reasoning_trace"]:
                result_text = "TRUE" if step["result"] else "FALSE"
                st.write(f"**{step['rule']}**")
                st.write(f"{step['condition']} → {result_text}")
            if d["decision"] == "SELECTED":
                st.success(f"Final Decision: {d['decision']}")
            else:
                st.error(f"Final Decision: {d['decision']}")


# ----------------------------------------------------------------------
# MODULE 9 - FINAL SEGMENTED IMAGE
# ----------------------------------------------------------------------
st.header("7. Final Segmented Image")

final_vis = pre["rgb"].copy()
selected_mask = np.zeros_like(final_mask)

if len(decisions) > 0:
    region_lookup = {int(r["region_id"]): r for _, r in regions_df.iterrows()}

    for d in decisions:
        rid = int(d["region_id"])
        r = region_lookup[rid]
        x, y, rw, rh = int(r["x"]), int(r["y"]), int(r["width"]), int(r["height"])

        if d["decision"] == "SELECTED":
            cv2.rectangle(final_vis, (x, y), (x + rw, y + rh), (0, 255, 0), 2)
            cv2.putText(final_vis, f"ID {rid}", (x, max(0, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Add this region's pixels to the final selected mask
            selected_mask[labels == rid] = 255

    st.image(final_vis, caption="Final Rule-Based Segmentation (green = AI-selected regions)",
              use_container_width=True)
    st.image(selected_mask, caption="Final Selected Regions Mask", use_container_width=True, clamp=True)
else:
    st.info("No final segmented image to display (no regions detected).")


# ----------------------------------------------------------------------
# MODULE - STATISTICS
# ----------------------------------------------------------------------
st.header("8. Statistics")

elapsed_time = round(time.time() - start_time, 4)
stats = summarize_decisions(decisions)

stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
stat_col1.metric("Total Regions", stats["total_regions"])
stat_col2.metric("Selected Regions", stats["selected_regions"])
stat_col3.metric("Rejected Regions", stats["rejected_regions"])
stat_col4.metric("Selection %", f"{stats['selection_percentage']}%")
stat_col5.metric("Processing Time", f"{elapsed_time}s")

st.caption(
    "Note: these are region-selection statistics produced by the rule "
    "engine, not a machine-learning accuracy score."
)


# ----------------------------------------------------------------------
# EVALUATION (OPTIONAL, REQUIRES GROUND TRUTH)
# ----------------------------------------------------------------------
st.header("9. Segmentation Evaluation (Optional)")

gt_mask_img = None
if gt_file is not None:
    try:
        gt_bytes = np.frombuffer(gt_file.read(), np.uint8)
        gt_mask_img = cv2.imdecode(gt_bytes, cv2.IMREAD_GRAYSCALE)
        if gt_mask_img is None:
            st.error("Could not read the ground-truth mask file.")
        elif gt_mask_img.shape != final_mask.shape:
            st.warning(
                f"Ground-truth mask shape {gt_mask_img.shape} does not match the "
                f"predicted mask shape {final_mask.shape}. Cannot evaluate."
            )
            gt_mask_img = None
    except Exception as e:
        st.error(f"Error reading ground-truth mask: {e}")
        gt_mask_img = None

eval_result = evaluate_segmentation(final_mask, gt_mask_img)

if eval_result["available"]:
    e1, e2, e3 = st.columns(3)
    e1.metric("IoU", eval_result["iou"])
    e2.metric("Dice Coefficient", eval_result["dice_coefficient"])
    e3.metric("Pixel Accuracy", eval_result["pixel_accuracy"])
else:
    st.info(eval_result["message"])


# ----------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------
st.markdown("---")
st.caption(
    "AI and Techniques Academic Project | Traditional AI (Knowledge-Based "
    "Rule Engine) + Classical Image Processing (OpenCV) | No ML / DL used."
)
