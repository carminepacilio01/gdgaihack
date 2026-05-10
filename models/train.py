# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from model import SmallPDTCN

# --- CONFIGURATION ---
FPS = 30
WINDOW_SEC = 4.0
WINDOW_FRAMES = int(FPS * WINDOW_SEC)
NUM_FRAME_FEATURES = 124  # Update based on your preprocessing logs
NUM_FFT_FEATURES = 24     # Update based on your preprocessing logs
EPOCHS = 50
BATCH_SIZE = 32
LR = 0.001

def load_data(npy_x_path, npy_meta_path):
    print("Loading data...")
    X = np.load(npy_x_path)
    meta = np.load(npy_meta_path)
    
    # Filter out NaNs
    labels = meta[:, 2]
    valid_idx = ~np.isnan(labels)
    X, labels = X[valid_idx], labels[valid_idx]
    
    # Split into Temporal and FFT
    split_idx = WINDOW_FRAMES * NUM_FRAME_FEATURES
    X_temporal = X[:, :split_idx].reshape(-1, WINDOW_FRAMES, NUM_FRAME_FEATURES)
    X_temporal = np.transpose(X_temporal, (0, 2, 1)) # Format: (B, C, T)
    X_fft = X[:, split_idx:]
    
    return X_temporal, X_fft, labels

if __name__ == "__main__":
    # 1. Load Data
    X_t, X_f, y = load_data("preprocessed/windows_X.npy", "preprocessed/windows_meta.npy")
    
    # 2. Split into Train and Validation (80/20)
    # Note: For clinical data, consider splitting by 'patient_id' using GroupShuffleSplit instead
    Xt_train, Xt_val, Xf_train, Xf_val, y_train, y_val = train_test_split(
        X_t, X_f, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to Tensors
    Xt_train, Xf_train, y_train = map(lambda x: torch.tensor(x, dtype=torch.float32), (Xt_train, Xf_train, y_train))
    Xt_val, Xf_val, y_val = map(lambda x: torch.tensor(x, dtype=torch.float32), (Xt_val, Xf_val, y_val))
    y_train, y_val = y_train.unsqueeze(1), y_val.unsqueeze(1)

    # 3. Initialize Model, Loss, Optimizer
    model = SmallPDTCN(NUM_FRAME_FEATURES, NUM_FFT_FEATURES)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 4. Training Loop
    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(Xt_train, Xf_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(Xt_val, Xf_val)
            val_loss = criterion(val_outputs, y_val)
            val_preds = (val_outputs >= 0.5).float()
            val_acc = (val_preds == y_val).float().mean()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {loss.item():.4f} - Val Loss: {val_loss.item():.4f} - Val Acc: {val_acc.item()*100:.2f}%")

    # 5. Save Model Checkpoint
    torch.save(model.state_dict(), "pd_tcn_weights.pth")
    print("\nModel weights saved to 'pd_tcn_weights.pth'")

    # 6. Export to ONNX for Luxonis Deployment
    model.eval()
    dummy_t = torch.randn(1, NUM_FRAME_FEATURES, WINDOW_FRAMES)
    dummy_f = torch.randn(1, NUM_FFT_FEATURES)
    
    torch.onnx.export(
        model, (dummy_t, dummy_f), "pd_tcn_model.onnx",
        input_names=["temporal_input", "fft_input"], output_names=["pd_probability"], opset_version=11
    )
    print("ONNX model exported to 'pd_tcn_model.onnx'")