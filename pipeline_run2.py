# # pipeline_run2.py
# import torch
# import torch.nn as nn
# from dataset_loader2 import get_splits_and_loaders
# from model2 import RobotGestureResNet

# def main():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print("=====================================================")
#     print("      PRODUCTION BASE GENERATIVE TRAINING ENGINE     ")
#     print("=====================================================\n")
    
#     MOTHER_PATH = "C:/Users/Tanmoy Paul/OneDrive/Desktop/model_2/datasets"

#     # 1. Load and Split Mother Dataset for rigorous evaluation
#     print(f"[Step 1] Loading Mother Training Dataset from: {MOTHER_PATH}")
#     mother_train_loader, mother_val_loader = get_splits_and_loaders(MOTHER_PATH, split_data=True)
#     num_classes = len(mother_train_loader.dataset.dataset.class_names)
    
#     # 2. Initialize Model Architecture
#     model = RobotGestureResNet(num_classes=num_classes).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
#     # 3. Train Mother Brain Base
#     print("\n[Step 2] Beginning Training Optimization Loop...")
#     for epoch in range(20): # Increased to 20 epochs for deeper generalization boundaries
#         model.train()
#         for matrices, labels in mother_train_loader:
#             matrices, labels = matrices.to(device), labels.to(device)
#             optimizer.zero_grad()
#             loss = criterion(model(matrices), labels)
#             loss.backward()
#             optimizer.step()
#         print(f"   Epoch [{epoch+1}/20] optimization completed.", end="\r")
            
#     # 4. Compute Final Evaluation Accuracy on Hidden Split
#     model.eval()
#     correct, total = 0, 0
#     with torch.no_grad():
#         for matrices, labels in mother_val_loader:
#             matrices, labels = matrices.to(device), labels.to(device)
#             _, predicted = torch.max(model(matrices), 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
            
#     true_accuracy = (correct / total) * 100
#     print(f"\n\n🔥 VERIFIED SYSTEM GENERALIZATION ACCURACY: {true_accuracy:.2f}%")
    
#     # Save the generalized model weights file cleanly
#     torch.save(model.state_dict(), 'mother_model_weights.pth')
#     print("Saved production weights to 'mother_model_weights.pth'\n")

# if __name__ == '__main__':
#     main()




import argparse
import torch
import torch.nn as nn
from dataset_loader2 import get_splits_and_loaders
from model2 import RobotGestureResNet

def main():
    # 1. Command-line argument parsing to choose hardware dynamically
    parser = argparse.ArgumentParser(description="Train RobotGestureResNet.")
    parser.add_argument(
        '--device', 
        type=str, 
        default='auto', 
        choices=['auto', 'cpu', 'cuda'],
        help="Hardware device to run on: 'cpu', 'cuda' (GPU), or 'auto' (automatic detection)"
    )
    args = parser.parse_args()

    # 2. Assign device based on user parameter
    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # Check if CUDA is actually available on the system
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            print("⚠️ WARNING: CUDA (GPU) requested but is not available on this machine. Falling back to CPU.\n")
    else:  # 'auto' mode
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=====================================================")
    print("      PRODUCTION BASE GENERATIVE TRAINING ENGINE     ")
    print(f"      TARGET HARDWARE DEVICE: {device.type.upper()}  ")
    print("=====================================================\n")
    
    MOTHER_PATH = "C:/Users/Tanmoy Paul/OneDrive/Desktop/model_2/datasets"

    # 3. Load and Split Mother Dataset for rigorous evaluation
    print(f"[Step 1] Loading Mother Training Dataset from: {MOTHER_PATH}")
    mother_train_loader, mother_val_loader = get_splits_and_loaders(MOTHER_PATH, split_data=True)
    num_classes = len(mother_train_loader.dataset.dataset.class_names)
    
    # 4. Initialize Model Architecture on the designated device
    model = RobotGestureResNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 5. Train Mother Brain Base
    print("\n[Step 2] Beginning Training Optimization Loop...")
    for epoch in range(20):  # 20 epochs for deeper generalization boundaries
        model.train()
        for matrices, labels in mother_train_loader:
            matrices, labels = matrices.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(matrices), labels)
            loss.backward()
            optimizer.step()
        print(f"   Epoch [{epoch+1}/20] optimization completed.", end="\r")
            
    # 6. Compute Final Evaluation Accuracy on Hidden Split
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for matrices, labels in mother_val_loader:
            matrices, labels = matrices.to(device), labels.to(device)
            _, predicted = torch.max(model(matrices), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    true_accuracy = (correct / total) * 100
    print(f"\n\n🔥 VERIFIED SYSTEM GENERALIZATION ACCURACY: {true_accuracy:.2f}%")
    
    # Save the generalized model weights file cleanly
    torch.save(model.state_dict(), 'mother_model_weights.pth')
    print("Saved production weights to 'mother_model_weights.pth'\n")

if __name__ == '__main__':
    main()