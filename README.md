<div align="center">
  
#  FractureScan: Automated Fracture Localization
</div>


<p align="center">
  <i>End-to-end inference stack for identifying fracture regions in diagnostic X-ray studies.</i>
</p>

##  Overview

FractureScan is a web-delivered inference pipeline built on FastAPI and Ultralytics YOLOv8. The backend accepts radiographs, performs fracture localization, and exposes Grad-CAM overlays so clinicians can trace the model's activation regions. Results, intermediate tensors, and Grad-CAM heatmaps are stored on disk for audit and downstream analysis.

##  Key Capabilities

- **Multiclass localization**: YOLOv8 head trained on transverse, oblique, spiral, comminuted, and greenstick labels
- **Low-latency inference**: Typical forward pass plus preprocessing completes within ~2 seconds on CPU
- **Explainability outputs**: Grad-CAM maps rendered per detection to highlight pixel regions influencing the logits
- **Structured metadata**: Confidence scores, bounding boxes, and class IDs returned in JSON for downstream systems
- **Static frontend**: HTML/JS client issues API calls, displays overlays, and surfaces Grad-CAM artifacts
- **REST interface**: FastAPI routes support automation and integration testing


##  Architecture

FractureScan exposes a FastAPI application that receives multipart uploads, writes them to `backend/uploads/`, and executes the YOLOv8 pipeline (preprocessing, inference, non-max suppression). Detection metadata plus Grad-CAM heatmaps are persisted under `backend/results/` to keep the request/response contract reproducible. The static frontend issues REST calls to `/detect`, polls for completion, and presents the rendered overlays.

1. **Backend (FastAPI)**: Async routes, Pydantic validation, PyTorch inference, Grad-CAM rendering, file persistence
2. **Frontend (HTML/CSS/JS)**: Static bundle that handles file selection, POSTs payloads, and displays bounding boxes plus Grad-CAM layers

##  Technology Stack

- **Backend runtime**: Python 3.8+, FastAPI, Uvicorn, Pydantic, PyTorch 2.x, Ultralytics YOLOv8, OpenCV
- **Frontend bundle**: HTML5 templates, vanilla JavaScript for API bindings, CSS3 for layout, FontAwesome icons
- **Model artifacts**: YOLOv8 weights fine-tuned on curated fracture datasets; configs live in `backend/models/`
- **Visualization**: Grad-CAM pipeline implemented via PyTorch hooks to produce overlay-ready heatmaps

##  Project Structure

```
├── graphs/
│   ├── epochs_vs_accuracy.png
│   └── confusion_matrices/
├── models/
│   ├── yolo_model.py
├── backend/                # FastAPI backend
│   ├── app.py              # Main application file
│   ├── requirements.txt    # Backend dependencies
│   ├── models/             # YOLOv8 model files
│   ├── results/            # Detection results
│   │   ├── explanations/   # Model explanation images
│   │   └── gradcam/        # Grad-CAM visualization images
│   └── uploads/            # Uploaded X-ray images
│
├── frontend/               # Static web frontend
│   ├── index.html          # Home page
│   ├── detect.html         # Detection page
│   ├── gradcam.html        # Grad-CAM explanation page
│   ├── about.html          # About page
│   ├── features.html       # Features page
│   ├── team.html           # Team page
│   ├── contact.html        # Contact page
│   ├── styles.css          # Main stylesheet
│   └── images/             # Frontend images
│
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

##  Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Modern web browser

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/JenilPanchal/fracturescan.git
   cd fracturescan
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the backend server**
   ```bash
   cd backend
   uvicorn app:app --reload
   ```

4. **Open the frontend**
   - Navigate to the `frontend` folder
   - Open `index.html` in your web browser

   OR

   - Serve the frontend using a simple HTTP server:
     ```bash
     cd frontend
     python -m http.server 8080
     ```
   - Open `http://localhost:8080` in your browser

##  API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Lightweight readiness probe |
| `/status` | GET | Returns uptime, model checksum, and storage usage |
| `/detect` | POST | Accepts `multipart/form-data` containing an X-ray image, returns detection metadata and resource IDs |
| `/gradcam/{image_id}` | GET | Streams the stored Grad-CAM PNG for a prior inference ID |

### Example API Usage

```python
import requests

# Upload an X-ray image for detection (multipart/form-data)
with open('xray.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/detect',
        files={'file': ('xray.jpg', f, 'image/jpeg')}
    )

response.raise_for_status()
result = response.json()

print(f"Detection ID: {result['detection_id']}")
for bbox in result['detections']:
    print(
        f"class={bbox['label']} "
        f"conf={bbox['confidence']:.3f} "
        f"xyxy={bbox['bbox']}"
    )
print(f"Result image path: {result['result_image']}")
print(f"Grad-CAM image path: {result['gradcam_image']}")
```

##  Development Setup

For developers who want to contribute to the project:

1. **Create a virtual environment**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

2. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run backend with debug mode**
   ```bash
   cd backend
   uvicorn app:app --reload --debug
   ```


##  Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework

---
