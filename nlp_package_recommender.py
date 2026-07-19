# ============================================
# 1. Import Libraries
# ============================================

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import re


# ============================================
# 2. Load Dataset
# ============================================

keras.utils.set_random_seed(42)

df = pd.read_csv("medical_tourism_dataset.csv")

# print(df.head())
# print(df.shape)
# print(df.info())


# ============================================
# 3. Data Cleaning
# ============================================

# Remove missing values
df = df.dropna()

# Remove duplicate rows
df = df.drop_duplicates()

# print(df.isnull().sum())
# print(df.shape)


# ============================================
# 4. Text Preprocessing
# ============================================

# Convert to lowercase
df["message"] = df["message"].str.lower()

# Remove extra spaces
df["message"] = df["message"].str.strip()
df["message"] = df["message"].str.replace(r"\s+", " ", regex=True)

# Remove punctuation
df["message"] = df["message"].apply(
    lambda x: re.sub(r"[^\w\s]", "", x)
)


# ============================================
# 5. Separate Features and Labels
# ============================================

X = df["message"]
y = df["package"]

# print(X.head())
# print(y.head())


# ============================================
# 6. Label Encoding
# ============================================

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# print(encoder.classes_)


# ============================================
# 7. Tokenization
# ============================================

tokenizer = Tokenizer(oov_token="<OOV>")

tokenizer.fit_on_texts(X)

# print(tokenizer.word_index)


# ============================================
# 8. Convert Text to Sequences
# ============================================

X = tokenizer.texts_to_sequences(X)

# print(X[:5])


# ============================================
# 9. Padding
# ============================================

X = pad_sequences(
    X,
    padding="post"
)

# print(X.shape)


# ============================================
# 10. Train / Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# print(X_train.shape)
# print(X_test.shape)


# ============================================
# 11. Build Neural Network
# ============================================

model = keras.Sequential([

    keras.layers.Embedding(
        input_dim=len(tokenizer.word_index) + 1,
        output_dim=32
    ),

    keras.layers.GlobalAveragePooling1D(),

    keras.layers.Dense(
        16,
        activation="relu"
    ),

    keras.layers.Dense(
        len(encoder.classes_),
        activation="softmax"
    )

])

model.build(input_shape=(None, X.shape[1]))

# model.summary()


# ============================================
# 12. Compile Model
# ============================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================
# 13. Train Model
# ============================================

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    validation_data=(X_test, y_test)
)


# ============================================
# 14. Predict New Message
# ============================================

new_text = [
    "I have severe eye pain and blurry vision"
]

new_sequence = tokenizer.texts_to_sequences(new_text)

new_sequence = pad_sequences(
    new_sequence,
    maxlen=X.shape[1],
    padding="post"
)

prediction = model.predict(new_sequence)

predicted_class = np.argmax(prediction)

package = encoder.inverse_transform([predicted_class])

print("Recommended Package:", package[0])

print("\nPrediction Probabilities:\n")

for label, prob in zip(encoder.classes_, prediction[0]):
    print(f"{label:<12}: {prob:.3f}")

print("\nRaw Prediction:")
print(np.round(prediction, 3))