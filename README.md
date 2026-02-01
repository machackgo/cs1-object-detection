---
title: CS1 Object Detection
emoji: 🧳
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: 5.6.1
app_file: app.py
pinned: false
license: mit
---

# CS1 Object Detection (DETR ResNet-50)

A simple object detection web app built with **Gradio** and the **Hugging Face Inference API**.

## Features
- Upload an image → get detected objects with bounding boxes
- Adjustable **score threshold**
- Adjustable **top_k** (max boxes shown)
- Clean, sorted detection table (highest confidence first)

## Run locally
```bash
pip install -r requirements.txt
python app.py
