---
title: CS1 Object Detection (DETR ResNet-50)
emoji: 🎯
colorFrom: green
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# CS1 Object Detection (DETR ResNet-50)

## Overview

A Gradio-based web application for object detection using **DETR ResNet-50**, a pretrained deep learning model that identifies and localizes objects in images. The application integrates with the **Hugging Face Inference API** for model inference, enabling fast, cloud-based object detection without requiring local GPU resources.

**Live Demo**: [Hugging Face Spaces](https://huggingface.co/spaces/MOHDMUBASHIR/cs1-object-detection)

## Problem Solved

Object detection is a fundamental computer vision task, but deploying production-grade detection systems requires managing large model files and compute resources. This project simplifies the workflow by:

- Leveraging Hugging Face's pretrained DETR model (no training required)
- - Using the Hugging Face Inference API to eliminate local model storage
  - - Providing an intuitive Gradio interface for image upload and result visualization
    - - Making object detection accessible through a simple web interface
     
      - ## Computer Vision Workflow
     
      - ```
        ┌─────────────────────┐
        │  User Uploads Image │
        │  (Upload Interface) │
        └──────────┬──────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │  Image Preprocessing     │
        │  (PIL Image Loading)     │
        └──────────┬───────────────┘
                   │
                   ▼
        ┌──────────────────────────────────────┐
        │  DETR ResNet-50 Inference            │
        │  (Hugging Face Inference API)        │
        │  Detects objects & bounding boxes    │
        └──────────┬───────────────────────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │  Post-Processing         │
        │  - Filter by threshold   │
        │  - Sort by confidence    │
        │  - Draw bounding boxes   │
        └──────────┬───────────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │  Output Visualization    │
        │  - Annotated image       │
        │  - Detection summary     │
        │  - Confidence scores     │
        └──────────────────────────┘
        ```

        ## Tech Stack

        - **Model**: DETR (Detection Transformer) ResNet-50 via Hugging Face
        - - **Inference**: Hugging Face Inference API (router.huggingface.co)
          - - **UI Framework**: Gradio 4.0+
            - - **Image Processing**: Pillow, requests
              - - **Language**: Python 3
                - - **Deployment**: Hugging Face Spaces
                 
                  - ## Key Features
                 
                  - - **Pretrained Model Inference**: Uses facebook/detr-resnet-50 model via Hugging Face Inference API
                    - - **Image Upload**: Accepts image files and processes them for object detection
                      - - **Bounding Box Detection**: Identifies objects and returns precise bounding box coordinates
                        - - **Confidence Scoring**: Each detection includes a confidence score
                          - - **Adjustable Detection Threshold**: Filter detections by minimum confidence score (0.0–1.0)
                            - - **Top-K Control**: Limit results to top N detected objects for clarity
                              - - **Sorted Results**: Detections automatically sorted by confidence (highest first)
                                - - **Detection Summary**: Text output shows object count and detection parameters used
                                  - - **Intuitive Gradio Interface**: No coding required—upload and analyze images instantly
                                   
                                    - ## How to Run Locally
                                   
                                    - ### Prerequisites
                                   
                                    - - Python 3.8+
                                      - - pip (Python package manager)
                                        - - Hugging Face API Token (obtain from [hf.co/settings/tokens](https://huggingface.co/settings/tokens))
                                         
                                          - ### Setup
                                         
                                          - 1. **Clone the repository**:
                                            2.    ```bash
                                                     git clone https://github.com/machackgo/cs1-object-detection.git
                                                     cd cs1-object-detection
                                                     ```

                                                  2. **Install dependencies**:
                                                  3.    ```bash
                                                           pip install -r requirements.txt
                                                           ```

                                                        3. **Set environment variable**:
                                                        4.    ```bash
                                                                 export HF_TOKEN=your_hugging_face_token_here
                                                                 ```

                                                              4. **Run the application**:
                                                              5.    ```bash
                                                                       python app.py
                                                                       ```

                                                                    5. **Access the interface**: Open your browser and navigate to `http://localhost:7860`
                                                                
                                                                    6. ### Requirements
                                                                
                                                                    7. Key dependencies (see `requirements.txt` for full list):
                                                                    8. - `gradio>=4.0.0` – Web UI framework
                                                                       - - `requests` – HTTP requests for API calls
                                                                         - - `pillow` – Image processing
                                                                           - - `pandas` – Data handling
                                                                            
                                                                             - ## Skills Demonstrated
                                                                            
                                                                             - ### Computer Vision & ML
                                                                             - - **Pretrained Model Integration**: Leveraging state-of-the-art DETR model without retraining
                                                                               - - **Inference Pipeline**: Building end-to-end inference from image input to visualization
                                                                                 - - **Image Processing**: Handling image loading, resizing, and annotation
                                                                                   - - **Object Detection**: Understanding bounding box representation and confidence scores
                                                                                    
                                                                                     - ### Software Engineering
                                                                                     - - **API Integration**: Connecting to Hugging Face Inference API with authentication
                                                                                       - - **Web Framework**: Building interactive UI with Gradio
                                                                                         - - **Parameter Control**: Implementing adjustable thresholds and filtering logic
                                                                                           - - **Error Handling**: Graceful handling of API calls and image processing
                                                                                            
                                                                                             - ### Full-Stack Development
                                                                                             - - **Backend Logic**: Processing images and coordinating API calls (app.py)
                                                                                               - - **Frontend Design**: Creating intuitive user interface for image upload and results display
                                                                                                 - - **Deployment**: Hosting on Hugging Face Spaces for public access
                                                                                                   - - **Configuration Management**: Managing API tokens and environment variables securely
                                                                                                    
                                                                                                     - ## VeriBridge Proof Evidence
                                                                                                    
                                                                                                     - | Capability | Evidence File | Location | Details |
                                                                                                     - |---|---|---|---|
                                                                                                     - | DETR ResNet-50 Inference | `app.py` | Line 1–15 | Model ID: `facebook/detr-resnet-50` loaded via Hugging Face Inference API |
                                                                                                     - | Hugging Face Inference API | `app.py` | Line 8–12 | API endpoint: `router.huggingface.co`; HF_TOKEN authentication |
                                                                                                     - | Image Upload Input | `app.py` | Line 20+ | Gradio interface accepts image input via `gr.Image()` |
                                                                                                     - | Bounding Box Drawing | `app.py` | `draw_boxes()` function | Draws rectangles and labels on detected objects |
                                                                                                     - | Adjustable Threshold | `app.py` | `score_threshold` slider | Parameter filters detections by confidence |
                                                                                                     - | Adjustable Top-K | `app.py` | `top_k` slider | Limits output to top N detections |
                                                                                                     - | Gradio Web UI | `app.py` | `gr.Interface()` | Full Gradio interface definition |
                                                                                                     - | Live Deployment | Hugging Face Spaces | [cs1-object-detection Space](https://huggingface.co/spaces/MOHDMUBASHIR/cs1-object-detection) | Active deployment with public access |
                                                                                                     - | Dependencies | `requirements.txt` | Lines 1–4 | gradio>=4.0.0, requests, pillow, pandas |
                                                                                                     - | License | Root directory | `LICENSE` | MIT License |
                                                                                                    
                                                                                                     - ## Recruiter Value
                                                                                                    
                                                                                                     - This project showcases:
                                                                                                    
                                                                                                     - 1. **Modern ML/CV Stack**: Hands-on experience with transformer-based object detection (DETR), a cutting-edge architecture
                                                                                                       2. 2. **API Integration**: Demonstrated ability to authenticate and integrate with third-party APIs (Hugging Face Inference API)
                                                                                                          3. 3. **Full-Stack Prototyping**: From inference pipeline to polished web UI—showing complete project execution
                                                                                                             4. 4. **Deployment & Scaling**: Published to Hugging Face Spaces, proving ability to ship production-grade applications
                                                                                                                5. 5. **Parameter Tuning**: Implements dynamic threshold and top-K controls, showing understanding of inference optimization
                                                                                                                   6. 6. **Clean Code Practices**: Well-organized Gradio interface with clear function structure
                                                                                                                     
                                                                                                                      7. Ideal for roles in: **Machine Learning Engineering**, **Computer Vision**, **Backend/Full-Stack Engineering**, **ML Ops**
                                                                                                                     
                                                                                                                      8. ## Future Improvements
                                                                                                                     
                                                                                                                      9. - **Batch Processing**: Support uploading and processing multiple images in parallel
                                                                                                                         - - **Webcam Input**: Add real-time object detection from webcam feeds
                                                                                                                           - - **Custom Model Support**: Allow users to select different DETR variants or other detection models
                                                                                                                             - - **Result Caching**: Cache results for identical images to reduce API calls
                                                                                                                               - - **Confidence Distribution Plot**: Visualize confidence score distribution across detections
                                                                                                                                 - - **Model Comparison**: Compare outputs from multiple detection models side-by-side
                                                                                                                                   - - **Fine-Tuning Option**: Allow users to fine-tune the model on custom datasets
                                                                                                                                     - - **Export Results**: Download annotated images or detection data as JSON/CSV
                                                                                                                                      
                                                                                                                                       - ---
                                                                                                                                       
                                                                                                                                       **Created by**: [machackgo](https://github.com/machackgo)
                                                                                                                                       **License**: MIT
