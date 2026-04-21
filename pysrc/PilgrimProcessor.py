

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import os
# ============================
# 1. LOAD AND PREPROCESS DATA
# ============================
# Save your full dataset as 'starlink_data.csv' (with many rows) in the same folder.
# The script expects the exact columns shown in your example.
print(Path.cwd())

# Check if data directory exists and move working directory up one level if it does not to find the data directory
if not Path('data').exists():
    print("Data directory not found in current working directory. Moving up one level to find data directory.")
    os.chdir('..')
print(Path.cwd())
data_path = Path('data/demo_data/starlink_data.csv')  # <-- CHANGE TO YOUR CSV FILE PATH
df = pd.read_csv(data_path) 

print(f"Loaded {len(df)} rows of data.")

# Convert boolean columns to 0/1
df['in_view'] = df['in_view'].astype(int)
df['fov'] = df['fov'].astype(int)          # Target: 0 = False, 1 = True

# Drop non-useful columns for the model (sat_name is string and unique per satellite)
# You can keep it if you one-hot encode, but for most datasets it's too high-cardinality
X = df.drop(columns=['fov', 'sat_name'])
y = df['fov']

# Encode any remaining categorical / ID columns (st_id, other_sat_id)
cat_cols = ['st_id', 'other_sat_id']
for col in cat_cols:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# Select numerical features for scaling
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Scale numerical features (critical for neural networks)
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Convert to numpy arrays
X_np = X.values.astype(np.float32)
y_np = y.values.astype(np.float32).reshape(-1, 1)

# Train / test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_np, y_np, test_size=0.2, random_state=42, stratify=y_np if len(np.unique(y_np)) > 1 else None
)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
print(f"Feature count: {X_train.shape[1]}")
# ============================
# 2. PYTORCH DATASET & DATALOADER
# ============================
class SatelliteDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = SatelliteDataset(X_train, y_train)
test_dataset = SatelliteDataset(X_test, y_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ============================
# 3. DEEP NEURAL NETWORK MODEL
# ============================
class FOVPredictor(nn.Module):
    def __init__(self, input_size):
        super(FOVPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1)          # Output logit for binary classification
        )
    
    def forward(self, x):
        return self.network(x)

input_dim = X_train.shape[1]
model = FOVPredictor(input_dim)

# ============================
# 4. TRAINING SETUP
# ============================
criterion = nn.BCEWithLogitsLoss()      # Best loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Training on: {device}")

# ============================
# 5. TRAINING LOOP
# ============================
epochs = 50
best_loss = float('inf')
patience = 10
counter = 0

print("\n=== Starting Training ===")
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    
    avg_val_loss = val_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy:.2f}%")
    
    # Early stopping
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_fov_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered!")
            break

print("\n✅ Training completed! Best model saved as 'best_fov_model.pth'")

# ============================
# 6. FINAL EVALUATION
# ============================
model.load_state_dict(torch.load('best_fov_model.pth'))
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

print(f"\nFinal Test Accuracy: {100 * correct / total:.2f}%")

# ============================
# 7. PREDICTION EXAMPLE (using your provided row)
# ============================
# Quick demo: predict on the single row you gave
example_data = {
    'st_id': [1036],
    'time': [1776560809.208741],
    'latitude': [48.12766669237274],
    'longitude': [-108.61317705027977],
    'altitude_km': [370.9640589324547],
    'footprint_radius_km': [65.4109724629168],
    'other_sat_id': [5395],
    'in_view': [0],                    # False → 0
    'fov_overlap_km': [151.48581662151256]
}

example_df = pd.DataFrame(example_data)

# Apply same preprocessing
for col in cat_cols:
    if col in example_df.columns:
        example_df[col] = le.transform(example_df[col].astype(str))  # use fitted encoder

example_df[numeric_cols] = scaler.transform(example_df[numeric_cols])

example_tensor = torch.tensor(example_df.values.astype(np.float32)).to(device)

model.eval()
with torch.no_grad():
    logit = model(example_tensor)
    prob = torch.sigmoid(logit).item()
    prediction = "True" if prob > 0.5 else "False"

print(f"\nPrediction on your example row → FOV = {prediction} (probability = {prob:.4f})")
