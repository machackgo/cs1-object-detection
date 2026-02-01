import os
import io
import requests
import gradio as gr
import pandas as pd
from PIL import Image, ImageDraw

# -------------------------
# Config
# -------------------------
# You can also set this as a HF Space variable named MODEL_ID if you want.
# (Settings -> Variables and secrets -> Variables)
MODEL_ID = os.getenv("MODEL_ID", "facebook/detr-resnet-50").strip()

# ✅ NEW: official router base (HF now prefers router.huggingface.co)
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"

# ✅ Token from HF Space secrets:
# Settings -> Variables and secrets -> Secrets -> add HF_TOKEN
HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip()

headers = {}
if HF_TOKEN:
    headers["Authorization"] = f"Bearer {HF_TOKEN}"


# -------------------------
# Utilities
# -------------------------
def draw_boxes(image: Image.Image, detections, score_threshold=0.50):
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    for d in detections:
        score = float(d.get("score", 0.0))
        if score < float(score_threshold):
            continue

        label = d.get("label", "object")
        box = d.get("box", {}) or {}

        xmin = int(box.get("xmin", 0))
        ymin = int(box.get("ymin", 0))
        xmax = int(box.get("xmax", 0))
        ymax = int(box.get("ymax", 0))

        # rectangle
        draw.rectangle([xmin, ymin, xmax, ymax], width=3)
        # label text
        draw.text((xmin, max(0, ymin - 12)), f"{label} {score:.2f}")

    return img


# -------------------------
# Main inference function
# -------------------------
def detect(image, threshold=0.50, top_k=50):
    """
    Professional outputs:
    - Sort detections by score (desc)
    - Apply top_k after sorting
    - Filter by threshold
    - Show summary line
    - Return pandas DataFrame
    """
    if image is None:
        return None, pd.DataFrame(), "", "No image uploaded."

    # Image -> bytes (PNG)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    try:
        # router requires a supported content-type (NOT octet-stream, NOT multipart)
        local_headers = dict(headers)
        local_headers["Content-Type"] = "image/png"

        r = requests.post(API_URL, headers=local_headers, data=img_bytes, timeout=60)

        # Model loading case
        if r.status_code == 503:
            return image, pd.DataFrame(), "", "Model is loading on HF. Wait 10–20 seconds and click Submit again."

        # Any non-200 error
        if r.status_code != 200:
            return image, pd.DataFrame(), "", f"HTTP {r.status_code}: {r.text}"

        detections = r.json()

        # HF can return {"error":"..."}
        if isinstance(detections, dict) and "error" in detections:
            return image, pd.DataFrame(), "", f"HF ERROR: {detections['error']}"

        if not isinstance(detections, list):
            return image, pd.DataFrame(), "", f"Unexpected response format: {type(detections)}"

        # 1) sort by score high -> low
        detections_sorted = sorted(
            detections,
            key=lambda d: float(d.get("score", 0.0)),
            reverse=True
        )

        # 2) apply top_k after sorting
        top_k = int(top_k)
        detections_topk = detections_sorted[:top_k]

        # 3) filter by threshold
        threshold = float(threshold)
        detections_filtered = [
            d for d in detections_topk if float(d.get("score", 0.0)) >= threshold
        ]

        # Draw boxes
        out_img = draw_boxes(image, detections_filtered, score_threshold=threshold)

        # Build table rows
        rows = []
        for d in detections_filtered:
            b = d.get("box", {}) or {}
            rows.append({
                "label": d.get("label", ""),
                "score": round(float(d.get("score", 0.0)), 4),
                "xmin": int(b.get("xmin", 0)),
                "ymin": int(b.get("ymin", 0)),
                "xmax": int(b.get("xmax", 0)),
                "ymax": int(b.get("ymax", 0)),
            })

        df = pd.DataFrame(rows, columns=["label", "score", "xmin", "ymin", "xmax", "ymax"])

        # Summary line
        summary = f"Found {len(detections_filtered)} objects (threshold={threshold:.2f}, top_k={top_k})"

        return out_img, df, summary, ""

    except Exception as e:
        return image, pd.DataFrame(), "", f"ERROR: {repr(e)}"


# -------------------------
# Gradio UI
# -------------------------
demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="pil", label="Upload image"),
        gr.Slider(0.0, 1.0, value=0.50, step=0.05, label="Score threshold"),
        gr.Slider(1, 100, value=50, step=1, label="Max boxes to show (top_k)"),
    ],
    outputs=[
        gr.Image(type="pil", label="Detections (boxed)"),
        gr.Dataframe(label="Detections (sorted + filtered)"),
        gr.Textbox(label="Summary"),
        gr.Textbox(label="Error / Info"),
    ],
    title="CS1 Object Detection (DETR ResNet-50)",
    description="Object detection demo using Hugging Face Inference API via router.huggingface.co",
)

if __name__ == "__main__":
    demo.launch()