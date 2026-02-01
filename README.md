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

A simple object detection web app built with **Gradio** and the **[Hugging Face](chatgpt://generic-entity?number=0) Inference API**.

## Features
- Upload an image → get detected objects with bounding boxes
- Adjustable **score threshold**
- Adjustable **top_k** (max boxes shown)
- Detections sorted by confidence (highest first)
- Summary line like: `Found N objects (threshold=..., top_k=...)`

## Live Demo
- Hugging Face Space: https://huggingface.co/spaces/MOHDMUBASHIR/cs1-object-detection

## Source Code
- GitHub Repo: https://github.com/machackgo/cs1-object-detection

## Run locally
```bash
pip install -r requirements.txt
python app.py
