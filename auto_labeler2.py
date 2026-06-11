# auto_labeler.py
import os
import shutil
import torch
import numpy as np
from model2 import RobotGestureResNet
from pose_extractor import extract_gesture_matrix_from_image

def run_auto_labeler(unlabelled_dir, output_labeled_dir, confidence_threshold=0.72):
    """
    Advanced Heuristic-Override Auto-Labeler Engine. Resolves spatial blindness
    by enforcing hard geometric checks on competing arm-to-head configurations.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Auto-Labeler Engine on: {device}")

    class_mapping = {
        0: 'OpenPalm', 1: 'ShoulderPain', 2: 'StomachPain', 
        3: 'ThumbDown', 4: 'ThumbUp', 5: 'headache', 6: 'neckpain'
    }

    num_classes = len(class_mapping)
    model = RobotGestureResNet(num_classes=num_classes).to(device)
    
    if not os.path.exists('mother_model_weights.pth'):
        print("❌ Error: 'mother_model_weights.pth' not found! Train your model first.")
        return
        
    model.load_state_dict(torch.load('mother_model_weights.pth', map_location=device))
    model.eval()

    for label_name in class_mapping.values():
        os.makedirs(os.path.join(output_labeled_dir, label_name), exist_ok=True)
    os.makedirs(os.path.join(output_labeled_dir, 'Uncertain_Review'), exist_ok=True)

    if not os.path.exists(unlabelled_dir):
        print(f"❌ Error: Unlabelled source folder '{unlabelled_dir}' does not exist.")
        return

    raw_files = [f for f in os.listdir(unlabelled_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(raw_files)} unlabelled images to process.")
    print(f"Setting baseline confidence threshold to: {confidence_threshold * 100:.1f}%")
    print("🛡️ Spatial Trajectory Override Rules: ACTIVE\n")

    success_count = 0
    uncertain_count = 0

    with torch.no_grad():
        for filename in raw_files:
            img_path = os.path.join(unlabelled_dir, filename)
            
            matrix = extract_gesture_matrix_from_image(img_path)
            
            # Extract left and right wrist Y metrics from the spatial feature vector
            lw_y = matrix[0][19]
            rw_y = matrix[0][21]
            highest_wrist_y = min(lw_y, rw_y) 
            
            matrix_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            outputs = model(matrix_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            top_probs, top_indices = torch.topk(probabilities, k=2)
            best_prob = top_probs[0].item()
            second_prob = top_probs[1].item()
            predicted_idx = top_indices[0].item()
            
            assigned_label = class_mapping[predicted_idx]
            decision_margin = best_prob - second_prob
            
            # 💡 CRITICAL GEOMETRIC OVERRIDE ENGINE
            # Rule 1: Fix the Neck Pain swallowed by ThumbDown Trap
            if assigned_label == 'ThumbDown' and highest_wrist_y < 0.2:
                assigned_label = 'neckpain'
                best_prob = 0.92 
                
            # Rule 2: TIGHTENED GEOMETRIC CALIBRATION
            elif assigned_label == 'ThumbUp':
                if highest_wrist_y < -0.38: # Higher threshold to isolate actual forehead touches
                    assigned_label = 'headache'
                elif -0.38 <= highest_wrist_y < -0.15: # Locked shoulder region bracket
                    assigned_label = 'ShoulderPain'

            is_valid = True
            if assigned_label in ['StomachPain', 'ThumbDown']:
                if best_prob < 0.85 or decision_margin < 0.35:
                    is_valid = False
            else:
                if best_prob < confidence_threshold:
                    is_valid = False

            if is_valid:
                dest_folder = os.path.join(output_labeled_dir, assigned_label)
                success_count += 1
                print(f"✓ Labeled '{filename}' as [{assigned_label}] ({best_prob*100:.1f}%)")
            else:
                dest_folder = os.path.join(output_labeled_dir, 'Uncertain_Review')
                uncertain_count += 1
                print(f"⚠️ Filtered '{filename}' ({best_prob*100:.1f}%) -> Sent to Review")
                
            shutil.copy(img_path, os.path.join(dest_folder, filename))

    print("\n=============================================")
    print("        AUTO-LABELING JOB COMPLETE           ")
    print("=============================================")
    print(f"Successfully Labeled : {success_count} images")
    print(f"Sent to Review       : {uncertain_count} images")
    print(f"All outputs saved to : {output_labeled_dir}")
    print("=============================================\n")

if __name__ == '__main__':
    UNLABELLED_SOURCE = "C:/Users/Tanmoy Paul/OneDrive/Desktop/model_2/user_dataset_of_others"
    LABELED_OUTPUT = "C:/Users/Tanmoy Paul/OneDrive/Desktop/model_2/automatically_labeled_dataset"
    
    run_auto_labeler(UNLABELLED_SOURCE, LABELED_OUTPUT, confidence_threshold=0.72)