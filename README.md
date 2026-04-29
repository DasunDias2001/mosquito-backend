# Mosquito Species Classification

A deep learning project for classifying dengue mosquito species (Aedes aegypti vs Aedes albopictus) using computer vision. This project includes both a training pipeline and a FastAPI backend for real-time predictions.

## Features

- 🦟 **Mosquito Classification**: Distinguish between Aedes aegypti and Aedes albopictus species
- 🧠 **Deep Learning Model**: TensorFlow/Keras-based CNN model
- 🚀 **FastAPI Backend**: REST API for real-time predictions
- 📊 **Complete Pipeline**: Data preparation, training, evaluation, and prediction
- 📈 **Model Monitoring**: TensorBoard integration for training visualization

## Project Structure

```
mosquito-classification/
├── backend/                    # FastAPI backend
│   ├── app.py                 # Main API application
│   ├── model_handler.py       # Model loading and prediction logic
│   ├── utils.py              # File handling utilities
│   └── requirements.txt      # Backend dependencies
├── src/                       # Training pipeline source code
│   ├── data_preparation.py   # Dataset splitting and preprocessing
│   ├── model_builder.py      # CNN model architecture
│   ├── train.py              # Training logic
│   ├── evaluate.py           # Model evaluation
│   └── predict.py            # Prediction utilities
├── models/                    # Saved models and checkpoints
├── data/                      # Processed datasets
├── results/                   # Evaluation results and plots
├── logs/                      # TensorBoard logs
├── uploads/                   # Temporary uploaded files (backend)
├── config.py                  # Configuration settings
├── run_pipeline.py           # Main pipeline controller
└── requirements.txt          # Main project dependencies
```

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.16+
- FastAPI and Uvicorn (for backend)

### Setup

1. **Clone the repository** (if applicable) and navigate to the project directory:
   ```bash
   cd mosquito-classification
   ```

2. **Install main dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install backend dependencies** (if running the API):
   ```bash
   cd backend
   pip install -r requirements.txt
   cd ..
   ```

## Usage

### Training Pipeline

The project includes a complete pipeline for training and evaluating the mosquito classification model.

#### Run Complete Pipeline

Train the model from scratch with data preparation, training, and evaluation:

```bash
python run_pipeline.py --mode full
```

**Note**: This may take 2-6 hours on CPU or 30-90 minutes on GPU.

#### Run Individual Steps

**Data Preparation**:
```bash
python run_pipeline.py --mode prepare
```

**Model Training**:
```bash
python run_pipeline.py --mode train
```

**Model Evaluation**:
```bash
python run_pipeline.py --mode evaluate
```

### Backend API

The FastAPI backend provides a REST API for real-time mosquito species classification.

#### Start the API Server

```bash
cd backend
python app.py
```

The server will start on `http://localhost:8000`

#### API Endpoints

**Health Check**:
```bash
GET /
```
Returns API status and model loading status.

**Model Information**:
```bash
GET /model-info
```
Returns information about the loaded model (classes, input size, etc.).

**Predict Mosquito Species**:
```bash
POST /predict
Content-Type: multipart/form-data

file: [image file]
```
Upload an image file (jpg, jpeg, png, bmp, tiff) to get species prediction.

**Example Response**:
```json
{
  "success": true,
  "predicted_class": "aegypti",
  "confidence": 0.94,
  "probabilities": {
    "aegypti": 0.94,
    "albopictus": 0.06
  },
  "message": "Predicted as AEGYPTI with 94.00% confidence"
}
```

#### Using the API

**With curl**:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/mosquito_image.jpg"
```

**With Python**:
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("mosquito_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### Prediction from Command Line

Predict individual images or batches using the training pipeline:

**Single Image**:
```bash
python run_pipeline.py --mode predict --image path/to/mosquito.jpg
```

**Batch Prediction**:
```bash
python run_pipeline.py --mode predict --folder path/to/images/
```

**Show Prediction Plot**:
```bash
python run_pipeline.py --mode predict --image path/to/mosquito.jpg --show
```

## Model Details

- **Architecture**: Convolutional Neural Network (CNN)
- **Input Size**: 224x224 pixels
- **Classes**: 2 (aegypti, albopictus)
- **Framework**: TensorFlow/Keras
- **Preprocessing**: Image resizing and normalization

## Dataset

The project expects a dataset structure like:
```
Dataset/
├── raw/
│   ├── aegypti/     # Aedes aegypti images
│   └── albopictus/  # Aedes albopictus images
├── train/           # Training split (created by pipeline)
├── test/            # Test split (created by pipeline)
└── val/             # Validation split (created by pipeline)
```

## Results and Monitoring

- **Models**: Saved in `models/saved_models/`
- **TensorBoard Logs**: Available in `logs/tensorboard/`
- **Evaluation Results**: Stored in `results/evaluation/`
- **Plots**: Generated in `results/plots/`

View training progress with TensorBoard:
```bash
tensorboard --logdir logs/tensorboard
```

## Configuration

Modify `config.py` to adjust:
- Dataset paths
- Model hyperparameters
- Training settings
- Directory paths

## Troubleshooting

**Common Issues**:

1. **Model not loading**: Ensure the model file exists in `models/saved_models/`
2. **CUDA errors**: Install CUDA-compatible TensorFlow version for GPU support
3. **Port already in use**: Change the port in `backend/app.py` (default: 8000)
4. **Memory errors**: Reduce batch size in `config.py` for lower memory systems

**Backend Issues**:
- Ensure all backend dependencies are installed
- Check that the model path in `backend/model_handler.py` is correct
- Verify upload directory permissions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.