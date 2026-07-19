# Medical Tourism Package Recommender

A Natural Language Processing (NLP) project that recommends the most suitable medical tourism package based on a patient's text description.

## Features

- Text preprocessing
- Label encoding
- Tokenization
- Sequence padding
- Embedding layer
- Neural network classification
- Medical package prediction

## Packages

- Cardiac
- Cosmetic
- Dental
- Eye
- Fertility
- Neurology
- Orthopedic

## Tech Stack

- Python
- TensorFlow
- Keras
- Pandas
- NumPy
- Scikit-learn

## Project Structure

```text
medical-tourism-package-recommender/
│
├── medical_tourism_dataset.csv
├── nlp_package_recommender.py
└── README.md
```

## Dataset

- 7 medical packages
- 28 text samples per package
- 196 total samples

## Example

**Input**

```text
I have severe eye pain and blurry vision
```

**Output**

```text
Recommended Package: Eye
```

## Run

```bash
pip install tensorflow pandas numpy scikit-learn

python nlp_package_recommender.py
```

## Future Improvements

- Increase dataset size
- Use LSTM/Bidirectional LSTM
- Save and load trained models
- Build a web interface
