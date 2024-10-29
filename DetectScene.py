import numpy as np
import logging
import pickle
import argparse
import cv2
import os 
from prepare_dataset import calculate_color_distance, calculate_structural_distance

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_svm_model(model_path):
    logging.info(f"Loading SVM model from {model_path}...")
    with open(model_path, 'rb') as f:
        svm_model = pickle.load(f)
    logging.info("Model loaded successfully.")
    return svm_model

def detect_transitions(svm, video_path, threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    transitions = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        color_dist = calculate_color_distance(prev_frame, frame)
        struct_dist = calculate_structural_distance(prev_frame, frame)
        prediction = svm.predict([[color_dist, struct_dist]])[0]
        if prediction == 1:
            transitions.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)  # 记录转换时间
        prev_frame = frame

    cap.release()
    return transitions

def parse_args():
    parser = argparse.ArgumentParser(description='Load and use trained SVM model for inference')
    parser.add_argument('--model_path', type=str, default='svm_model.pkl', help='Path to the trained SVM model')
    parser.add_argument('--data_path', type=str, default='prepared_dataset.pkl', help='Path to the input dataset for inference')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    model_path = args.model_path
    data_path = args.data_path

    # Load the trained SVM model
    svm_model = load_svm_model(model_path)
    video_List = [os.path.join(data_path, video) for video in os.listdir(data_path) if video.endswith('mp4')]

    for video in video_List:
        transitions = detect_transitions(svm_model, video_path=video)
        print(f"{video} Detected transitions (in seconds):", transitions)
