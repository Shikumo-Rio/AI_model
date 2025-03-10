import pandas as pd
import tensorflow as tf
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import numpy as np
from flask import Flask, jsonify
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Global variable for model
model = None

def train_model():
    try:
        global model
        # Load dataset
        df = pd.read_csv('tasks.csv')

        # Convert text labels to numbers
        label_encoder = LabelEncoder()
        df['request'] = label_encoder.fit_transform(df['request'])  # Convert task type into numbers

        # Tokenize 'details' for text processing
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(df['details'])
        X_text = tokenizer.texts_to_sequences(df['details'])
        X_padded = pad_sequences(X_text, maxlen=10)

        # Define inputs (X) and outputs (y)
        X = np.hstack((df[['request', 'room']].values, X_padded))  # Combine request, room, and details
        y = np.where(df['status'] == 'Pending', 1, 0)  # 1 = Needs allocation, 0 = Already assigned

        # Build AI Model
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=1000, output_dim=16, input_length=X.shape[1]),
            tf.keras.layers.LSTM(16),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X, y, epochs=10, batch_size=4)
        model.save('task_allocation_model.keras')
        logger.info("✅ Model training complete and saved!")
        return True
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return False

@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "Hello, Task Allocation AI!"})

@app.route("/train", methods=['POST'])
def trigger_training():
    success = train_model()
    if success:
        return jsonify({"status": "success", "message": "Model training completed"})
    return jsonify({"status": "error", "message": "Model training failed"}), 500

if __name__ == "__main__":
    train_model()  # Initial training
    app.run(host='0.0.0.0', port=5000, debug=False)
