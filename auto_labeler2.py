# auto_labeler2.py
import os
import shutil
import torch
import numpy as np
from model2 import RobotGestureResNet
from pose_extractor import extract_gesture_matrix_from_image

def run_auto_labeler(unlabelled_dir, output_labeled_dir, chosen_gesture, confidence_threshold=0.72):
    """
    Advanced Heuristic-Override Auto-Labeler Engine -- ISOLATED SINGLE GESTURE MODE.
    Processes the source folder but ONLY saves and reports metrics for the chosen target.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Auto-Labeler Engine on: {device}")

    # 🟢 CHANGED: Replaced Unknown_Class with chestpain at index 7
    class_mapping = {
        0: 'OpenPalm', 1: 'ShoulderPain', 2: 'StomachPain', 
        3: 'ThumbDown', 4: 'ThumbUp', 5: 'headache', 6: 'neckpain',
        7: 'chestpain'
    }

    # Safety check to make sure your typing matches the system folders exactly
    if chosen_gesture not in class_mapping.values():
        print(f"❌ Error: '{chosen_gesture}' is not a valid gesture class!")
        print(f"Valid choices are: {list(class_mapping.values())}")
        return

    num_classes = len(class_mapping)
    model = RobotGestureResNet(num_classes=num_classes).to(device)
    
    if not os.path.exists('mother_model_weights.pth'):
        print("❌ Error: 'mother_model_weights.pth' not found! Train your model first.")
        return
        
    model.load_state_dict(torch.load('mother_model_weights.pth', map_location=device))
    model.eval()

    # ONLY create directories for our target and review folder to keep it clean
    os.makedirs(os.path.join(output_labeled_dir, chosen_gesture), exist_ok=True)
    os.makedirs(os.path.join(output_labeled_dir, 'Uncertain_Review'), exist_ok=True)

    if not os.path.exists(unlabelled_dir):
        print(f"❌ Error: Unlabelled source folder '{unlabelled_dir}' does not exist.")
        return

    raw_files = [f for f in os.listdir(unlabelled_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_input_images = len(raw_files)
    
    print(f"Found {total_input_images} raw images inside source folder.")
    print(f"🎯 TARGET ISOLATION FILTER ACTIVATED FOR: [{chosen_gesture}]")
    print(f"Setting baseline confidence threshold to: {confidence_threshold * 100:.1f}%\n")

    success_count = 0
    uncertain_count = 0
    total_times_detected = 0

    with torch.no_grad():
        for filename in raw_files:
            img_path = os.path.join(unlabelled_dir, filename)
            matrix = extract_gesture_matrix_from_image(img_path)
            
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
            
            # Heuristic Overrides
            if assigned_label == 'ThumbDown' and highest_wrist_y < 0.2:
                assigned_label = 'neckpain'
                best_prob = 0.92 
            elif assigned_label == 'ThumbUp':
                if highest_wrist_y < -0.38: 
                    assigned_label = 'headache'
                elif -0.38 <= highest_wrist_y < -0.15: 
                    assigned_label = 'ShoulderPain'

            # 💡 THE SINGLE GESTURE FILTER: If it is not our chosen target, skip it completely!
            if assigned_label != chosen_gesture:
                continue
                
            # If we reached here, it means the model detected our target gesture
            total_times_detected += 1

            # Check threshold constraints
            is_valid = True
            
            # 🟢 CHANGED: Added chestpain here to enforce strict safety margins
            if assigned_label in ['StomachPain', 'ThumbDown', 'chestpain']:
                if best_prob < 0.85 or decision_margin < 0.35:
                    is_valid = False
            else:
                if best_prob < confidence_threshold:
                    is_valid = False

            if is_valid:
                dest_folder = os.path.join(output_labeled_dir, assigned_label)
                success_count += 1
                print(f"✓ Labeled '{filename}' as [{assigned_label}] ({best_prob*100:.1f}%)")
                shutil.copy(img_path, os.path.join(dest_folder, filename))
            else:
                dest_folder = os.path.join(output_labeled_dir, 'Uncertain_Review')
                uncertain_count += 1
                print(f"⚠️ Filtered '{filename}' ({best_prob*100:.1f}%) -> Sent to Review")
                shutil.copy(img_path, os.path.join(dest_folder, filename))

    # --- 📊 SINGLE GESTURE PERFORMANCE DASHBOARD ---
    print("\n=======================================================")
    print(f"        ISOLATED REPORT FOR: {chosen_gesture.upper()}        ")
    print("=======================================================")
    print(f"Total Times detected by AI Network   : {total_times_detected} frames")
    print(f"Passed Confidence Gate (Clean)       : {success_count} images")
    print(f"Failed Confidence Gate (In Review)   : {uncertain_count} images")
    print("-------------------------------------------------------")
    
    yield_rate = (success_count / total_times_detected * 100) if total_times_detected > 0 else 0.0
    print(f"🎯 ACCURACY AUTOMATION YIELD RATE    : {yield_rate:.1f}%")
    print("=======================================================\n")

if __name__ == '__main__':
    # Forward slashes applied to point to your unlabelled frames directory
    UNLABELLED_SOURCE = "C:/Users/Tanmoy Paul/OneDrive/Desktop/model_2/all_pat"
    LABELED_OUTPUT = "C:/Users/Tanmoy Paul/OneDrive/Desktop/model_2/automatically_labeled_dataset"
    
    # 💡 CHANGE THIS STR TO CHECK A DIFFERENT GESTURE ONE-BY-ONE ('chestpain', 'neckpain', etc.)
    CHOSEN_GESTURE = 'chestpain' 
    
    run_auto_labeler(UNLABELLED_SOURCE, LABELED_OUTPUT, CHOSEN_GESTURE, confidence_threshold=0.72)