import numpy as np
import logging
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_svm(output_path):
    logging.info("Loading dataset for training...")
    with open(output_path, 'rb') as f:
        color_distances, struct_distances, labels = pickle.load(f)

    logging.info("Starting training of SVM model...")
    X = np.column_stack((color_distances, struct_distances))
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1, random_state=42)
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Training complete. Model accuracy: {accuracy}")
    return svm

def parse_args():
    parser = argparse.ArgumentParser(description='Train SVM model')
    parser.add_argument('--data_path', type=str, default='prepared_dataset.pkl', help='Path to the output dataset')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    video_path = args.data_path
    svm_model = train_svm(video_path)
    # Save the trained SVM model
    with open('svm_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)
    logging.info("SVM model saved to svm_model.pkl")
    