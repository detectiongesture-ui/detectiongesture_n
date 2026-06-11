# pose_extractor.py
import numpy as np
import cv2
from ultralytics import YOLO
import config

# Load core tracking layer
pose_model = YOLO('yolov8n-pose.pt')

def extract_gesture_matrix_from_image(image_path_or_frame, training_mode=False):
    # Accept both file paths or direct webcam numpy frame matrices
    if isinstance(image_path_or_frame, str):
        frame = cv2.imread(image_path_or_frame)
    else:
        frame = image_path_or_frame

    # 34 structural coordinates + 6 distance features = 40 features total
    if frame is None:
        return np.zeros((config.MAX_FRAMES, 40), dtype=np.float32)
        
    results = pose_model(frame, verbose=False)
    sequence_data = np.zeros((config.MAX_FRAMES, 40), dtype=np.float32)
    
    if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        kp = results[0].keypoints.xy[0].cpu().numpy()  # Shape: (17, 2)
        
        if kp.shape[0] == 17:
            left_shoulder, right_shoulder = kp[5], kp[6]
            left_wrist, right_wrist = kp[9], kp[10]
            nose = kp[0]
            left_hip, right_hip = kp[11], kp[12]
            
            neck_anchor = (left_shoulder + right_shoulder) / 2.0
            shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
            if shoulder_width == 0: 
                shoulder_width = 1.0
                
            # 1. Calculate relative coordinates
            normalized_kp = (kp - neck_anchor) / shoulder_width
            flat_coords = normalized_kp.flatten()  # 34 elements
            
            # Data Augmentation Noise Injection to break facial/body memorization
            if training_mode:
                temporal_noise = np.random.normal(0, 0.03, flat_coords.shape)
                flat_coords += temporal_noise
            
            # Scale coordinates strictly between -1.0 and 1.0 to keep spatial proportions
            coord_max = np.max(np.abs(flat_coords)) if np.max(np.abs(flat_coords)) > 0 else 1.0
            scaled_coords = flat_coords / coord_max
            
            # 2. Calculate explicit distance probes
            lw_to_nose = np.linalg.norm(left_wrist - nose) / shoulder_width
            rw_to_nose = np.linalg.norm(right_wrist - nose) / shoulder_width
            lw_to_neck = np.linalg.norm(left_wrist - neck_anchor) / shoulder_width
            rw_to_neck = np.linalg.norm(right_wrist - neck_anchor) / shoulder_width
            lw_to_hip  = np.linalg.norm(left_wrist - (left_hip + right_hip)/2.0) / shoulder_width
            rw_to_hip  = np.linalg.norm(right_wrist - (left_hip + right_hip)/2.0) / shoulder_width
            
            raw_distances = np.array([lw_to_nose, rw_to_nose, lw_to_neck, rw_to_neck, lw_to_hip, rw_to_hip])
            
            # Normalize distance dimensions independently so they don't drown out the coordinates
            dist_norm = np.linalg.norm(raw_distances) if np.linalg.norm(raw_distances) > 0 else 1.0
            scaled_distances = raw_distances / dist_norm
            
            # Combine both feature sets safely (Total 40 features)
            final_features = np.concatenate([scaled_coords, scaled_distances])
            
            # Distribute uniform tracking signatures across the timeline rows
            sequence_data[:] = final_features

    return sequence_data.astype(np.float32)