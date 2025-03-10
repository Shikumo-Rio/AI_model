import pandas as pd
import tensorflow as tf
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import numpy as np

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

# Train Model
model.fit(X, y, epochs=10, batch_size=4)

# Save Model
model.save('task_allocation_model.keras')
print("✅ Model training complete and saved!")
