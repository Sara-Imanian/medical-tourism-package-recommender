from flask import Flask, request, jsonify

from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

import joblib 
import numpy as np 

app = Flask(__name__)

model = keras.models.load_model("medical_package_model.keras")
tokenizer = joblib.load("tokenizer.pkl")
encoder = joblib.load("encoder.pkl")

@app.route("/")
def home():
    return "Medical Tourism API is running!"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()
    message = data["message"]
    sequence = tokenizer.texts_to_sequences([message])

    sequence = pad_sequences(
        sequence, 
        maxlen=model.input_shape[1],
        padding="post"
    )

    prediction = model.predict(sequence)
    predicted_class = np.argmax(prediction)
    package = encoder.inverse_transform([predicted_class])[0]

    return jsonify({
        "package": package
    })


print("Model loaded successfully!")

if __name__ == "__main__":
    app.run(debug=True)

