import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import os

# ============================
# 1. GPU SETUP
# ============================
print("=== NVIDIA GPU Check ===")
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("⚠️  Using CPU")

# ============================
# 2. LOAD DATA
# ============================
print(Path.cwd())

# Check if data directory exists and move working directory up one level if it does not to find the data directory
if not Path('data').exists():
    print("Data directory not found in current working directory. Moving up one level to find data directory.")
    os.chdir('..')
print(Path.cwd())
data_path = Path('data/demo_data/starlink_data.csv')  # <-- CHANGE TO YOUR CSV FILE PATH
df = pd.read_csv(data_path) 

print(f"\nLoaded {len(df)} rows.")
state_dict = None
model_path = Path('best_fov_model_imbalanced.pth')
if model_path.exists():
    print(f"✅ Found existing model at '{model_path}'. Loading model...")   
    state_dict = torch.load(model_path, map_location=device)

df['in_view'] = df['in_view'].astype(int)
df['fov'] = df['fov'].astype(int)

print("\nClass Distribution:")
print(df['fov'].value_counts())
print(f"Positive rate: {df['fov'].mean()*100:.2f}%")

X = df.drop(columns=['fov', 'sat_name'])
y = df['fov']

# Encode categorical
cat_cols = ['st_id', 'other_sat_id']
label_encoders = {}
for col in cat_cols:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

# Scale
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

X_np = X.values.astype(np.float32)
y_np = y.values.astype(np.float32).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X_np, y_np, test_size=0.2, random_state=42, stratify=y_np
)

# ============================
# 3. IMBALANCE HANDLING
# ============================
pos_count = y_train.sum()
neg_count = len(y_train) - pos_count
pos_weight_value = neg_count / (pos_count + 1e-8) # Multiply by 3 to give more emphasis to the minority class
pos_weight_value *= 10
print(f"\npos_weight: {pos_weight_value:.2f}")

# Weighted sampler
class_sample_count = np.array([neg_count, pos_count])
weights = 1. / class_sample_count
sample_weights = weights[y_train.astype(int).flatten()]

sampler = WeightedRandomSampler(
    torch.DoubleTensor(sample_weights), 
    num_samples=len(sample_weights), 
    replacement=True
)

# ============================
# 4. DATASET & DATALOADER
# ============================
class SatelliteDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 64
train_dataset = SatelliteDataset(X_train, y_train)
test_dataset = SatelliteDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# ============================
# 5. MODEL
# ============================
class FOVPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.network(x)

model = FOVPredictor(X_train.shape[1]).to(device)
if state_dict is not None:
    model.load_state_dict(state_dict)

# ============================
# 6. LOSS + OPTIMIZER
# ============================
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# ============================
# 7. TRAINING WITH METRIC TRACKING
# ============================
epochs = 2
history = {
    'train_loss': [],
    'val_loss': [],
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': []
}

best_loss = float('inf')
patience = 10
counter = 0

print("\n=== Starting Training ===\n")

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        
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
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    avg_val_loss = val_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    
    # Save history
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['accuracy'].append(accuracy)
    history['precision'].append(precision)
    history['recall'].append(recall)
    history['f1'].append(f1)
    
    print(f"Epoch [{epoch+1:2d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
          f"Acc: {accuracy:.3f} | F1: {f1:.4f} | P: {precision:.4f} | R: {recall:.4f}")
    
    # Early stopping
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_fov_model_imbalanced.pth')
        print("✅ Model saved as 'best_fov_model_imbalanced.pth'")  
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered!")
            break

print("\n✅ Training completed!")

# ============================
# 8. PLOTTING METRICS WITH MATPLOTLIB
# ============================
epochs_range = range(1, len(history['train_loss']) + 1)

plt.figure(figsize=(15, 10))

# Loss Plot
plt.subplot(2, 2, 1)
plt.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o')
plt.plot(epochs_range, history['val_loss'], label='Val Loss', marker='s')
plt.title('Train vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy
plt.subplot(2, 2, 2)
plt.plot(epochs_range, history['accuracy'], label='Accuracy', color='green', marker='o')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# F1 Score
plt.subplot(2, 2, 3)
plt.plot(epochs_range, history['f1'], label='F1 Score', color='blue', marker='o')
plt.title('Validation F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True)

# Precision & Recall
plt.subplot(2, 2, 4)
plt.plot(epochs_range, history['precision'], label='Precision', color='orange', marker='o')
plt.plot(epochs_range, history['recall'], label='Recall', color='red', marker='s')
plt.title('Precision and Recall')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('fov_training_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

print("📊 Training metrics plots saved as 'fov_training_metrics.png'")

# ============================
# 9. FINAL EVALUATION + EXAMPLE
# ============================
model.load_state_dict(torch.load('best_fov_model_imbalanced.pth', map_location=device))
model.eval()

print("\nFinal Test Results:")
all_preds = []
all_labels = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_y.numpy())

acc = accuracy_score(all_labels, all_preds)
p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {p:.4f}")
print(f"Recall   : {r:.4f}")
print(f"F1 Score : {f1:.4f}")

# Example prediction
print("\n--- Prediction on your example row ---")
# Load 50 sample data for prediction from file data/training_data/train_data.csv
example = pd.read_csv('data/training_data/train_data.csv').iloc[0:50]

for col in cat_cols:
    if col in example.columns:
        example[col] = label_encoders[col].transform(example[col].astype(str))

example[numeric_cols] = scaler.transform(example[numeric_cols])

example_tensor = torch.tensor(example.values.astype(np.float32)).to(device)

with torch.no_grad():
    logit = model(example_tensor)
    prob = torch.sigmoid(logit).item()
    pred = "True" if prob > 0.5 else "False"

print(f"Predicted FOV: {pred} (probability = {prob:.4f})")