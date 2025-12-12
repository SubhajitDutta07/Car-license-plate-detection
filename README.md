# License Plate Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)

A professional license plate detection and recognition system using YOLOv8 and OCR technologies. This project provides an easy-to-use web interface for detecting and extracting license plate text from images and videos.

## Features

- 🚗 **License Plate Detection**: Accurate detection using YOLOv8 object detection model
- 🔤 **OCR Integration**: Text recognition with EasyOCR and Tesseract OCR engines
- 📁 **Multiple Formats**: Support for images (JPG, PNG) and videos (MP4, AVI)
- 🌐 **Web Interface**: User-friendly Streamlit application for easy interaction
- 📊 **Results Export**: Save detection results in JSON format for further analysis
- ⚙️ **Configurable**: Adjustable confidence thresholds and OCR method selection
- 📈 **Real-time Processing**: Fast inference for real-time applications
- 🔧 **Modular Architecture**: Clean separation of data processing, model training, and inference

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download the project**:
   ```bash
   # If using git
   git clone <repository-url>
   cd license-plate-detection
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required models**:
   - Place your trained YOLOv8 model as `models/model.pt`
   - Ensure Tesseract OCR is installed (see below)

### Installing Tesseract OCR

#### macOS:
```bash
brew install tesseract
```

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

#### Windows:
Download from: https://github.com/UB-Mannheim/tesseract/wiki

## Usage

### Running the Web Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to the displayed URL (usually `http://localhost:8501`)

3. **Upload an image or video** and click "Detect License Plates"

### Command Line Usage

You can also use the individual modules:

```python
from src.inference import LicensePlateDetector

detector = LicensePlateDetector()
results = detector.detect_in_image("path/to/image.jpg", "output/path.jpg")
```

## Configuration

### Model Configuration

- Place your trained YOLOv8 model in the `models/` directory as `model.pt`
- Adjust confidence thresholds in the Streamlit interface

### Dataset Configuration

Configure your datasets in `config/datasets.yaml`:

```yaml
train:
  path: "path/to/train/data"
  format: "yolo"  # or "coco", "voc"

val:
  path: "path/to/val/data"
  format: "yolo"
```

## Project Structure

```
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── config/
│   └── datasets.yaml      # Dataset configuration
├── models/
│   └── model.pt          # Trained YOLOv8 model
├── src/
│   ├── data_preprocessing.py  # Data preparation utilities
│   ├── inference.py          # Inference and detection logic
│   ├── model_training.py     # Model training scripts
│   └── ocr_processor.py      # OCR text extraction
├── output/               # Detection results and exports
└── temp/                 # Temporary files
```

## Training Your Own Model

1. **Prepare your dataset** in YOLO format
2. **Configure datasets.yaml** with your data paths
3. **Run training**:
   ```bash
   python src/model_training.py
   ```

## API Reference

### LicensePlateDetector Class

- `detect_in_image(image_path, output_path)`: Detect plates in a single image
- `detect_in_video(video_path, output_path)`: Process video files
- `extract_text(image, method='easyocr')`: Extract text using OCR

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Troubleshooting

### Common Issues

1. **Model not loading**: Ensure `models/model.pt` exists and is a valid YOLOv8 model
2. **OCR not working**: Install Tesseract OCR and ensure it's in your PATH
3. **CUDA errors**: Install PyTorch with CUDA support if using GPU

### Performance Tips

- Use GPU for faster inference
- Adjust image size (imgsz) based on your hardware
- Lower confidence threshold for more detections (may increase false positives)

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection model
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for OCR capabilities
- [Streamlit](https://streamlit.io/) for the web interface