import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Example function to build the CNN
def build_cnn_model(input_length, num_classes):
    model = models.Sequential()
    model.add(layers.Embedding(input_dim=5000, output_dim=128, input_length=input_length))
    model.add(layers.Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='sigmoid'))  # Use sigmoid for multi-label classification
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Preprocessing function for text data
def preprocess_text_data(algQuestions, max_len):
    # Extract titles and content, combine them
    texts = ["Title: " + question.Title + "\n" + question.Content for question in algQuestions]
    target_labels = [question.Topics for question in algQuestions]

    # Tokenize the texts
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # Pad the sequences for uniform input size
    padded_sequences = pad_sequences(sequences, maxlen=max_len)

    # Binarize the multi-label targets
    mlb = MultiLabelBinarizer()
    target_labels = mlb.fit_transform(target_labels)

    return padded_sequences, target_labels, tokenizer, mlb

# Main training function
def train_cnn_model(algQuestions, recordPath, max_len=200, test_size=0.2, epochs=10, batch_size=32):
    # Preprocess the text and labels
    X, y, tokenizer, mlb = preprocess_text_data(algQuestions, max_len)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Get number of classes for output
    num_classes = y_train.shape[1]

    # Build the CNN model
    cnn_model = build_cnn_model(input_length=max_len, num_classes=num_classes)

    # Train the model
    history = cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                            epochs=epochs, batch_size=batch_size)

    # Save the model
    cnn_model.save(f"{recordPath}/cnn_model.h5")

    # Save tokenizer and MultiLabelBinarizer
    with open(f"{recordPath}/tokenizer.json", 'w') as f:
        json.dump(tokenizer.to_json(), f)

    with open(f"{recordPath}/mlb_classes.json", 'w') as f:
        json.dump(mlb.classes_.tolist(), f)

    return history

# Example usage
algQuestions = GetAllQuestions()  # Fetch algorithm questions
recordPath = "your_record_path_here"  # Define your record path

history = train_cnn_model(algQuestions, recordPath, max_len=200, epochs=10)
