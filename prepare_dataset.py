import cv2
import numpy as np
import logging
import os
import pickle
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_color_distance(frame1, frame2):
    hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    return np.dot(hist1, hist2)

def calculate_structural_distance(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    edge1 = cv2.Canny(gray1, 100, 200)
    edge2 = cv2.Canny(gray2, 100, 200)
    return np.mean((edge1 - edge2) ** 2)

def prepare_dataset(video_paths, output_path, num_samples=100, skip_start_frames=5, skip_end_frames=5, frame_distance=5):
    color_distances = []
    struct_distances = []
    labels = []
    
    for path_idx, path in enumerate(video_paths):
        logging.info(f"Processing video {path_idx + 1}/{len(video_paths)}: {path}")
        cap = cv2.VideoCapture(path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for i in range(skip_start_frames):
            cap.grab() 
            
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            if len(frames) >= total_frames - skip_start_frames - skip_end_frames:
                break

        cap.release()
        
        for _ in range(num_samples):
            idx1 = np.random.randint(0, len(frames) - frame_distance)
            idx2 = idx1 + np.random.randint(1, frame_distance)
            color_distances.append(calculate_color_distance(frames[idx1], frames[idx2]))
            struct_distances.append(calculate_structural_distance(frames[idx1], frames[idx2]))
            labels.append(0)
        
        for other_path in video_paths:
            if other_path == path:
                continue
            cap_other = cv2.VideoCapture(other_path)
            ret, frame_other = cap_other.read()
            cap_other.release()
            idx = np.random.choice(len(frames))
            color_distances.append(calculate_color_distance(frames[idx], frame_other))
            struct_distances.append(calculate_structural_distance(frames[idx], frame_other))
            labels.append(1)

    logging.info("Finished preparing the dataset")
    
    # 保存数据
    with open(output_path, 'wb') as f:
        pickle.dump((color_distances, struct_distances, labels), f)

def parse_args():
    parse_args = argparse.ArgumentParser(description='Prepare dataset for SVM')
    parse_args.add_argument('--video_path', type=str, required=True, help='Path to the directory containing videos')
    parse_args.add_argument('--output_path', type=str, required=True, help='Path to save the prepared dataset')
    parse_args.add_argument('--num_samples', type=int, default=100, help='Number of samples to generate for each video')
    args = parse_args.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    video_paths = args.video_paths
    output_path = args.output_path
    num_samples = args.num_samples

    video_List = [os.path.join(video_paths, video) for video in os.listdir(video_paths) if video.endswith('mp4')]
    
    prepare_dataset(video_List, output_path, num_samples=num_samples)
    print("Dataset prepared successfully!")