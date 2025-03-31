import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import re

class SentimentModelEvaluation:
    def __init__(self, model_path='models/advanced_sentiment_model.h5', dataset_path='twitter_training.csv'):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.model = None
        self.tokenizer = None
        self.label_encoder = LabelEncoder()
        self.max_length = 100
        
        # Load and prepare data
        self.prepare_data()
        
        # Load or train model
        self.load_or_train_model()
        
    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        # Remove special characters but keep emojis
        text = re.sub(r'[^\w\s\u263a-\U0001f645]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def prepare_data(self):
        """Prepare dataset for model evaluation"""
        # Load and prepare data
        df = pd.read_csv(self.dataset_path, encoding='utf-8', on_bad_lines='skip')
        df.columns = ["ID", "Category", "Sentiment", "Tweet"]
        df = df[["Sentiment", "Tweet"]].dropna()
        
        # Preprocess texts
        self.texts = df['Tweet'].apply(self.preprocess_text).tolist()
        self.labels = df['Sentiment'].tolist()
        
        # Encode labels
        self.label_encoder.fit(self.labels)
        self.encoded_labels = self.label_encoder.transform(self.labels)
        self.categorical_labels = tf.keras.utils.to_categorical(self.encoded_labels)
        
    def load_or_train_model(self):
        """Load existing model or train new one"""
        try:
            if os.path.exists(self.model_path):
                print("Loading existing model...")
                self.model = tf.keras.models.load_model(self.model_path)
                print("Model loaded successfully")
            else:
                self.train_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            self.train_model()
        
    def train_model(self):
        """Train the sentiment classification model"""
        # Prepare text sequences
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=5000, 
            oov_token="<OOV>"
        )
        self.tokenizer.fit_on_texts(self.texts)
        sequences = self.tokenizer.texts_to_sequences(self.texts)
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, 
            maxlen=self.max_length,
            padding='post',
            truncating='post'
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            padded_sequences,
            self.categorical_labels,
            test_size=0.2,
            random_state=42
        )
        
        # Store test data for later evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        # Build model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32),
            tf.keras.layers.Embedding(5000, 128, input_length=self.max_length),
            tf.keras.layers.SpatialDropout1D(0.2),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(len(self.label_encoder.classes_), activation='softmax')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = self.model.fit(
            X_train,
            y_train,
            epochs=10,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Save model
        self.model.save(self.model_path)
        
        return history
 

    def evaluate_model(self):
       y_pred = self.model.predict(self.X_test)
       y_pred_classes = y_pred.argmax(axis=1)  # Convert probabilities to class labels
       y_true = self.y_test.argmax(axis=1)  # True labels

    # Calculate metrics
       accuracy = accuracy_score(y_true, y_pred_classes)
       precision = precision_score(y_true, y_pred_classes, average='weighted')
       recall = recall_score(y_true, y_pred_classes, average='weighted')
       f1 = f1_score(y_true, y_pred_classes, average='weighted')

    # Create a table to display metrics
       metrics_table = pd.DataFrame({
          'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        '  Score': [accuracy, precision, recall, f1]
       })

       print("\nModel Performance Metrics:")
       print(metrics_table)

    
    def plot_confusion_matrix(self, conf_matrix):
        """Create and save confusion matrix visualization"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()

# Usage
evaluator = SentimentModelEvaluation()
results = evaluator.evaluate_model()

# Print results
print("\n--- Model Performance Metrics ---")
print("\nOverall Metrics:")
for metric, value in results['metrics'].items():
    print(f"{metric.replace('_', ' ').title()}: {value:.4f}")

print("\nDetailed Classification Report:")
for class_name, metrics in results['classification_report'].items():
    if isinstance(metrics, dict):
        print(f"\n{class_name.upper()} Class:")
        for metric, value in metrics.items():
            print(f"  {metric.title()}: {value:.4f}")


if __name__ == "__main__":
    evaluator = SentimentModelEvaluation()
    results = evaluator.evaluate_model()
