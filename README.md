# Automatic License Plate Recognition (ALPR)

A complete end-to-end solution for detecting and recognizing license plates using deep learning.

## Features

- **Data Preparation**: XML annotation parsing and preprocessing
- **Model Training**: InceptionResNetV2-based object detection
- **Flask Web App**: User-friendly interface for uploading and processing images
- **OCR Integration**: Tesseract-based text extraction
- **YOLOv5 Pipeline**: Alternative YOLO-based detection pipeline

## Project Structure

```
Number-Plate-Recognition/
├── data_preparation.py       # Data loading and preprocessing
├── keras_pipeline.py         # Model training and inference
├── flask_app.py             # Flask web application
├── yolov5_pipeline.py       # YOLOv5 alternative pipeline
├── config.py                # Configuration settings
├── requirements.txt         # Python dependencies
├── templates/
│   └── index.html          # Flask HTML template
├── static/
│   ├── upload/             # Uploaded images
│   ├── roi/                # Extracted license plates
│   └── predict/            # Detection results
└── models/                 # Trained models
```

## Requirements

- Python 3.8+
- Tesseract OCR (install separately)
- CUDA (optional, for GPU support)

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/BadalSharma007/Number-Plate-Recognition.git
cd Number-Plate-Recognition
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install Tesseract** (if not already installed):
- **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
- **macOS**: `brew install tesseract`
- **Windows**: Download from [here](https://github.com/UB-Mannheim/tesseract/wiki)

## Usage

### 1. Data Preparation

```python
from data_preparation import parse_xml_annotations, preprocess_data, split_data
from config import ANNOTATIONS_PATH, IMAGES_PATH, DATA_PATH

# Parse annotations
df = parse_xml_annotations(ANNOTATIONS_PATH)
X, y = preprocess_data(df, IMAGES_PATH)
x_train, x_test, y_train, y_test = split_data(X, y)
```

### 2. Training

```python
from keras_pipeline import build_model, train_model, save_model
from config import KERAS_MODEL_H5_PATH

# Build and train model
model = build_model()
history = train_model(model, x_train, y_train, x_test, y_test)
save_model(model, KERAS_MODEL_H5_PATH)
```

### 3. Flask Web Application

```bash
python flask_app.py
# Open browser to http://localhost:5000
```

### 4. API Usage

```python
import requests

with open('test_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/predict',
        files={'image': f}
    )
    print(response.json())
```

## Model Performance

- **Architecture**: InceptionResNetV2 (Transfer Learning)
- **Input Size**: 224x224
- **Loss Function**: MSE
- **Optimizer**: Adam (lr=1e-4)
- **Epochs**: 500
- **Batch Size**: 10

## Results

Sample results show successful detection and OCR of license plates with high accuracy.

## Future Improvements

- [ ] Integrate YOLOv5 pipeline
- [ ] Add confidence scores
- [ ] Batch processing support
- [ ] Azure cloud deployment
- [ ] Model quantization for edge devices

## License

MIT License - see LICENSE file for details

## Author

Badal Sharma

## Contact

- GitHub: [@BadalSharma007](https://github.com/BadalSharma007)

