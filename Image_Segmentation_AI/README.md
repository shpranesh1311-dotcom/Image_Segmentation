# AI-Based Image Segmentation Using Rule-Based and Traditional Image Processing Techniques

**Subject:** AI and Techniques
**Syllabus Topic:** Image Recognition — Image Segmentation
**Program:** B.Tech, Artificial Intelligence and Data Science

---

## 1. Project Title

AI-Based Image Segmentation Using Rule-Based and Traditional Image Processing Techniques

## 2. Abstract

This project demonstrates image segmentation using **classical digital image processing** techniques (thresholding, edge detection, morphology, connected-component analysis) combined with a **traditional, symbolic AI reasoning layer** — a knowledge-based rule engine. After an image is segmented and candidate regions are detected, the system extracts simple geometric features (area, width, height, aspect ratio) for each region. A rule engine then applies explicit **IF-THEN rules**, using a configurable **knowledge base** of thresholds, to reason about which regions are relevant and which should be discarded. The project deliberately avoids machine learning and deep learning, so that the "AI" demonstrated is the classical, explainable, rule-based style of AI taught in traditional AI courses (expert systems, knowledge representation, forward chaining).

## 3. Introduction

Image segmentation is a fundamental step in image recognition. It partitions an image into meaningful regions so that further analysis (feature extraction, object identification) becomes possible. While modern approaches often rely on machine learning or deep learning models, **traditional AI** relies on symbolic reasoning: explicit rules operating over facts derived from data. This project builds a small but complete pipeline that performs segmentation using OpenCV and then hands the extracted region features to a rule-based reasoning engine, which is the "AI and Techniques" component being demonstrated.

## 4. Problem Statement

Given an input image, segment it into distinct regions using classical image processing methods, extract measurable features from each detected region, and use a knowledge-based rule engine to decide — through explicit, explainable IF-THEN logic — which regions are relevant, without using any machine learning, deep learning, or pretrained models.

## 5. Objectives

1. Implement multiple classical image segmentation techniques.
2. Perform connected-component-based region detection.
3. Extract geometric features from each region.
4. Build a knowledge base of thresholds that represents domain knowledge.
5. Implement a forward-chaining-style rule engine that reasons over region features.
6. Visually and textually demonstrate the reasoning process (facts → rules → decision).
7. Provide an interactive Streamlit interface for experimentation.
8. Avoid any use of machine learning or deep learning.

## 6. Scope

The project covers segmentation of single 2D images (JPG/PNG) using thresholding, edge-based and region-based classical techniques, followed by rule-based filtering of detected regions. It does not cover video segmentation, 3D data, semantic labelling of object classes, or any form of learned/statistical classification.

## 7. Existing System

Most modern image segmentation systems (e.g., U-Net, Mask R-CNN, YOLO-based segmentation) rely on deep neural networks trained on large labelled datasets. These systems are highly accurate but are "black boxes" — their internal decision-making is difficult to explain, they require large amounts of training data and compute, and they cannot be easily audited rule-by-rule.

## 8. Proposed System

The proposed system uses **classical, deterministic image processing** for segmentation (no training required) and a **transparent, rule-based AI layer** for decision-making. Every decision the system makes can be traced back to a specific, human-readable rule (e.g., "area >= 500 → TRUE"). This makes the system fully explainable, lightweight, and suitable for teaching traditional AI concepts such as knowledge representation and forward chaining.

## 9. System Architecture

```
Input Image
     |
Preprocessing (Grayscale, Gaussian Blur)
     |
Image Segmentation (Thresholding / Edge / Region-based)
     |
Region Detection (Connected-Component Analysis)
     |
Feature Extraction (Area, Width, Height, Aspect Ratio, Centroid)
     |
Knowledge Base (minimum_area, minimum_width, minimum_height)
     |
Rule-Based Reasoning Engine (Forward-Chaining IF-THEN Rules)
     |
Decision (SELECTED / REJECTED per region)
     |
Final Segmented Image
```

## 10. Methodology

1. **Input:** The user uploads an image (or uses the built-in sample) through the Streamlit UI.
2. **Preprocessing:** The image is converted to grayscale and optionally smoothed with a Gaussian blur to reduce noise.
3. **Segmentation:** One of five classical segmentation methods converts the grayscale image into a binary mask.
4. **Region Detection:** `cv2.connectedComponentsWithStats()` identifies distinct foreground regions.
5. **Feature Extraction:** For every region, area, width, height, bounding box, centroid, and aspect ratio are computed.
6. **Knowledge Base:** The user configures `minimum_area`, `minimum_width`, and `minimum_height` via the sidebar.
7. **Rule Engine:** Each region's features are evaluated against the knowledge base using explicit IF-THEN rules, in a forward-chaining style.
8. **Decision & Output:** Regions that satisfy all rules are marked SELECTED and highlighted in the final image; others are REJECTED.

## 11. Traditional AI Concepts

This project is built around **symbolic / traditional AI**, not statistical learning:

- **Knowledge Representation:** Region features (area, width, height) are represented as *facts*.
- **Knowledge Base:** A small set of configurable thresholds represents domain expertise.
- **Rules:** IF-THEN rules encode the logic for deciding relevance.
- **Inference Engine:** The rule engine matches facts against rules to derive conclusions.
- **Forward Chaining:** Facts are evaluated rule-by-rule, and intermediate conclusions (`area_condition`, `width_condition`, `height_condition`) are combined to reach the final decision — the classic "data-driven" inference strategy used in expert systems.

## 12. Image Processing Techniques

- **Grayscale Conversion** — reduces the image to a single intensity channel.
- **Gaussian Blur** — smooths the image to reduce high-frequency noise before segmentation.
- **Binary / Otsu / Adaptive Thresholding** — converts grayscale images into binary foreground/background masks.
- **Canny Edge Detection** — detects object boundaries based on intensity gradients.
- **Connected-Component Analysis** — groups neighbouring foreground pixels into distinct labelled regions.

(Edge-based and region-based segmentation use a small amount of morphology internally to close broken edges and merge nearby fragments into solid regions — this is part of how those two segmentation algorithms work, not a separate processing stage.)

None of these techniques involve learning from data; all are deterministic mathematical/algorithmic operations, which is why OpenCV alone is **not** considered "AI" in this project.

## 13. Rule Engine

Implemented in `rule_engine.py`. For every detected region, the engine evaluates:

```
RULE 1: IF area   >= minimum_area   THEN area_condition   = TRUE
RULE 2: IF width  >= minimum_width  THEN width_condition  = TRUE
RULE 3: IF height >= minimum_height THEN height_condition = TRUE
RULE 4: IF area_condition AND width_condition AND height_condition
        THEN region = SELECTED
        ELSE region = REJECTED
```

The engine returns, for every region, the value of every intermediate condition plus the final decision — this trace is displayed directly in the Streamlit UI under "AI Rule-Based Decision".

## 14. Knowledge Base

```python
knowledge_base = {
    "minimum_area": 500,
    "minimum_width": 20,
    "minimum_height": 20,
}
```

These values are **not hard-coded** — they are configurable from the Streamlit sidebar, allowing the user to see how changing the knowledge base changes which regions are selected, without touching any code.

## 15. Forward-Chaining Reasoning

Forward chaining starts from known **facts** (region area, width, height) and applies rules repeatedly until no more rules can fire, arriving at a final conclusion. In this project:

```
Facts:      area = 850, width = 40, height = 30
Rule 1 fires:  area >= 500  → area_condition = TRUE
Rule 2 fires:  width >= 20  → width_condition = TRUE
Rule 3 fires:  height >= 20 → height_condition = TRUE
Rule 4 fires:  all TRUE     → region = SELECTED
```

This is a simplified, teaching-oriented version of forward chaining as used in expert systems such as CLIPS or MYCIN.

## 16. Feature Extraction

Implemented in `feature_extraction.py` using `cv2.connectedComponentsWithStats()`. For every region (excluding the background label `0`):

| Feature | Description |
|---|---|
| Region ID | Unique label assigned by connected-component analysis |
| Area | Number of foreground pixels in the region |
| Width / Height | Bounding box dimensions |
| Bounding Box (x, y, w, h) | Top-left corner and size |
| Centroid (cx, cy) | Center of mass of the region |
| Aspect Ratio | width / height |

## 17. Project Structure

```
Image_Segmentation_AI/
│
├── app.py                  # Streamlit UI and pipeline orchestration
├── preprocessing.py         # Grayscale conversion, Gaussian blur
├── segmentation.py          # 5 classical segmentation methods
├── feature_extraction.py    # Connected-component analysis & features
├── rule_engine.py           # Knowledge base + IF-THEN rule engine (THE AI)
├── evaluation.py            # IoU / Dice / Pixel Accuracy (optional, needs GT mask)
├── generate_sample.py       # Creates the built-in sample image
├── requirements.txt
├── README.md
│
├── input_images/
│   └── sample.jpg
│
├── output/
│   ├── mask.png
│   ├── segmented.png
│   └── result.png
│
└── screenshots/
    └── README.txt
```

## 18. Installation

```bash
cd Image_Segmentation_AI
pip install -r requirements.txt
```

If a sample image is not already present:

```bash
python generate_sample.py
```

## 19. Execution

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints in your browser (usually `http://localhost:8501`).

## 20. Expected Output

- Original and preprocessed images displayed side by side.
- A binary segmentation mask for the chosen method.
- A table of all detected regions with their features.
- A step-by-step "AI Rule-Based Decision" trace for every region.
- A final image with green bounding boxes around AI-selected regions only.
- Statistics: total / selected / rejected regions, selection percentage, and processing time.
- Optional IoU / Dice / Pixel Accuracy if a ground-truth mask is uploaded.

## 21. Advantages

- Fully explainable — every decision can be traced to a specific rule.
- No training data or GPU required.
- Fast and lightweight.
- Easy to modify rules/knowledge base without retraining anything.
- Good teaching tool for both classical image processing and symbolic AI.

## 22. Limitations

- Cannot recognize object *categories* (e.g., "this is a car") — it only reasons about geometric properties.
- Rule thresholds must be manually tuned per use case; they do not adapt automatically.
- Performance depends heavily on image contrast and lighting.
- Not robust to complex, cluttered, real-world scenes the way learned models can be.

## 23. Applications

- Educational demonstrations of classical AI and image processing.
- Simple quality-control filters (e.g., rejecting very small defects/regions by size).
- Document/shape analysis where objects are well separated from the background.
- Prototyping rule-based logic before considering a learned approach.

## 24. Future Enhancements

- Add more rules (e.g., shape circularity, solidity, perimeter).
- Support batch processing of multiple images.
- Add a rule-editor UI so users can add/remove rules without editing code.
- Extend the knowledge base to support fuzzy (rather than crisp) thresholds.

## 25. Conclusion

This project demonstrates that meaningful, explainable AI-driven decisions can be made without machine learning or deep learning. By combining classical image processing (OpenCV) with a symbolic knowledge-based rule engine, the system segments images and reasons about which regions matter — clearly separating "image processing" (OpenCV) from "AI" (the rule engine), exactly as required by the syllabus topic of traditional AI and techniques.

## 26. Viva Questions and Answers

**1. What is image segmentation?**
The process of dividing an image into multiple meaningful regions or segments, typically to separate objects from the background.

**2. Why is image segmentation important?**
It is a key pre-processing step for further image analysis, feature extraction, and recognition tasks.

**3. What are the types of image segmentation?**
Threshold-based, edge-based, region-based, and (in modern systems) learning-based segmentation. This project uses only the first three, classical types.

**4. What is thresholding?**
A technique that converts a grayscale image into a binary image by classifying pixels as foreground or background based on an intensity threshold.

**5. What is Otsu thresholding?**
An automatic thresholding method that calculates the optimal threshold value by minimizing intra-class intensity variance.

**6. What is adaptive thresholding?**
A thresholding method that computes a different threshold for each local region of the image, useful when lighting is uneven.

**7. What is edge-based segmentation?**
Segmentation based on detecting sharp changes in intensity (edges), typically using operators like Canny, followed by joining edges into closed boundaries.

**8. What is region-based segmentation?**
Segmentation that groups pixels into regions based on similarity criteria (e.g., intensity), here implemented classically via thresholding plus morphological merging — not machine-learning clustering.

**9. What is connected-component analysis?**
An algorithm that scans a binary image and assigns a unique label to each group of connected foreground pixels, allowing individual regions to be identified and measured.

**10. What is morphological processing?**
A set of operations (erosion, dilation, opening, closing) that process images based on shapes, typically used to remove noise or fill gaps in binary masks.

**11. What is feature extraction?**
The process of computing measurable properties (e.g., area, width, height, aspect ratio) from detected regions, which can then be used for further reasoning or decision-making.

**12. What is traditional AI?**
Symbolic, rule-based AI that represents knowledge explicitly (facts and rules) and reasons using logical inference, as opposed to statistical/machine learning approaches that learn patterns from data.

**13. What is a knowledge base?**
A structured collection of facts and rules that represents domain knowledge used by an AI system to make decisions — in this project, the minimum area/width/height thresholds.

**14. What is a rule engine?**
A software component that evaluates a set of IF-THEN rules against known facts to derive conclusions or decisions.

**15. What is an IF-THEN rule?**
A conditional statement of the form "IF condition THEN action/conclusion", the fundamental building block of rule-based expert systems.

**16. What is forward chaining?**
An inference strategy that starts from known facts and applies matching rules repeatedly to derive new facts/conclusions, continuing until a final decision is reached.

**17. Where is AI used in this project?**
In `rule_engine.py` — the knowledge base and IF-THEN rule evaluation that decides whether each segmented region is SELECTED or REJECTED. OpenCV is used only for classical image processing, not for the AI decision.

**18. Why are you not using machine learning?**
The project topic is "Traditional AI and Techniques," which specifically covers symbolic, rule-based reasoning rather than statistical learning. Using ML/DL would not demonstrate the syllabus concepts of knowledge representation and rule-based inference.

**19. What are the limitations?**
The system cannot recognize object categories, needs manually tuned thresholds, and performs best on images with clear foreground/background contrast.

**20. What are future enhancements?**
Adding more sophisticated geometric rules (circularity, solidity), a rule editor UI, batch processing, and fuzzy-logic-based thresholds.

---

### Academic Note

OpenCV is used for classical image processing operations such as grayscale conversion, thresholding, edge detection, morphology, and connected-component analysis. The traditional AI component is implemented through a knowledge-based rule engine that reasons over extracted region features and makes explicit IF-THEN decisions. The system performs segmentation using classical image processing and applies traditional AI reasoning to determine which segmented regions satisfy the defined knowledge-based rules. OpenCV itself is not an AI model, and the system does not "predict" objects using AI — it reasons over measured features using explicit rules.
