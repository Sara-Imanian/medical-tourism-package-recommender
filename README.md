# Medical Tourism Package Recommender

A Natural Language Processing (NLP) project that predicts the most appropriate medical tourism package based on a patient's medical description using TensorFlow and Keras.

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
- Confidence score for each predicted package

## Tech Stack

- Python
- TensorFlow / Keras
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
├── medical_tourism_dataset.csv
├── nlp_package_recommender.py
└── README.md
```

## Example

**Input**

```text
I have severe eye pain and blurry vision.
```

**Output**

```text
Recommended Package: Eye

Cardiac     : 0.136
Cosmetic    : 0.097
Dental      : 0.146
Eye          : 0.225
Fertility   : 0.135
Neurology   : 0.120
Orthopedic  : 0.141
```

## Installation

```bash
pip install tensorflow pandas numpy scikit-learn
```

## Run

```bash
python nlp_package_recommender.py
```

## Future Improvements

- Expand the dataset
- Improve text preprocessing
- Experiment with LSTM and Bidirectional LSTM
- Deploy as a web application
