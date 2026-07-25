# Medical Tourism Package Recommender

A Natural Language Processing (NLP) project that predicts the most appropriate medical tourism package from a patient's medical description using TensorFlow, Keras, and a Flask REST API.

## Available Medical Packages

- Cardiac
- Cosmetic
- Dental
- Eye
- Fertility
- Neurology
- Orthopedic

## Features

- Medical text preprocessing
- Multi-class text classification
- Word embedding with Keras Embedding layer
- Automatic package recommendation
- Prediction confidence scores
- Flask REST API

## Tech Stack

- Python
- TensorFlow / Keras
- Flask
- Scikit-learn
- Pandas
- NumPy

## Dataset

- 420 medical text samples
- 7 medical packages
- 60 samples per package

## Project Structure

```text
medical-tourism-package-recommender/
│
├── app.py
├── train.py
├── medical_package_model.keras
├── tokenizer.pkl
├── encoder.pkl
├── medical_tourism_dataset.csv
├── requirements.txt
└── README.md
```

## API Example

### Request

```http
POST /predict
```

```json
{
    "message": "I have severe eye pain and blurry vision"
}
```

### Response

```json
{
    "package": "Eye"
}
```

## Installation

```bash
pip install -r requirements.txt
```

## Train Model

```bash
python train.py
```

## Run API

```bash
python app.py
```

## Future Improvements

- Expand the dataset
- Improve text preprocessing
- Experiment with LSTM and Bidirectional LSTM
- Docker support
