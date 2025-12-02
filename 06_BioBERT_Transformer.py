import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np

# ---------------------- Modular Functions ---------------------- #

def load_data(file_path):
    """Load and preprocess the dataset."""
    df = pd.read_csv(file_path)
    df = df[['Claim_Line_ID', 'bio_text', 'Label']].dropna()
    return df

def get_embeddings(texts, model_name="dmis-lab/biobert-base-cased-v1.1"):
    """Generate BioBERT embeddings using LangChain."""
    embedder = HuggingFaceEmbeddings(model_name=model_name)
    embeddings = embedder.embed_documents(texts)
    return np.array(embeddings)

def train_classifier(X_train, y_train):
    """Train a Logistic Regression classifier."""
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(y_true, y_pred):
    """Evaluate model performance using multiple metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'classification_report': classification_report(y_true, y_pred)
    }
    return metrics

def save_predictions(ids, y_true, y_pred, output_file):
    """Save actual and predicted labels to a CSV."""
    result_df = pd.DataFrame({
        'Claim_Line_ID': ids,
        'Actual_Label': y_true,
        'Predicted_Label': y_pred
    })
    result_df.to_csv(output_file, index=False)

# ---------------------- Main Workflow ---------------------- #

if __name__ == "__main__":
    # Load data
    data = load_data(os.path.join("Data", "biobert_upcoding_features.csv"))

    # Split data
    X_train_texts, X_test_texts, y_train, y_test, id_test = train_test_split(
        data['bio_text'], data['Label'], data['Claim_Line_ID'], test_size=0.2, random_state=42
    )

    # Generate embeddings
    print("Generating embeddings for training set...")
    X_train_embeddings = get_embeddings(X_train_texts.tolist())
    print("Generating embeddings for test set...")
    X_test_embeddings = get_embeddings(X_test_texts.tolist())

    # Train classifier
    print("Training classifier...")
    model = train_classifier(X_train_embeddings, y_train)

    # Predict
    print("Predicting on test set...")
    y_pred = model.predict(X_test_embeddings)

    # Evaluate
    metrics = evaluate_model(y_test, y_pred)
    print("Model Evaluation Metrics:")
    for key, value in metrics.items():
        if key != 'classification_report':
            print(f"{key}: {value}")
    print("\nClassification Report:\n", metrics['classification_report'])

    # Save predictions
    save_predictions(id_test.tolist(), y_test.tolist(), y_pred.tolist(), os.path.join("Data", "upcoding_predictions.csv"))
    print("Predictions saved to upcoding_predictions.csv")