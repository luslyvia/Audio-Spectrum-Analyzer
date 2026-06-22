import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add path to the resnet1d directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'resnet1d'))
from resnet1d.resnet1d import ResNet1D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load and Preprocess Data
print("Loading dataset...")
df = pd.read_csv('submarine_dataset.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Stratified Split 80/20 to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Save Train and Test sets for reference
train_df = pd.DataFrame(X_train)
train_df['Label'] = y_train
train_df.to_csv('train_set.csv', index=False)

test_df = pd.DataFrame(X_test)
test_df['Label'] = y_test
test_df.to_csv('test_set.csv', index=False)
print("Saved 'train_set.csv' and 'test_set.csv'")

# Normalization (StandardScaler) to improve convergence
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class SubmarineDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

train_dataset = SubmarineDataset(X_train, y_train)
test_dataset = SubmarineDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 2. Initialize Model (Using architecture described in the paper)
# Tuning: Increased base_filters to 64 to learn deeper patterns, keeping dropout enabled (use_do=True uses p=0.5 in source code).
model = ResNet1D(
    in_channels=1, base_filters=64, kernel_size=5, stride=2, groups=1, n_block=3, n_classes=5, use_bn=True, use_do=True
).to(device)

epochs = 100
# Tuning: Reduced Learning Rate to 0.0005 for finer steps, and adjusted weight decay
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True)

# Using standard CrossEntropyLoss (Removed class_weights to optimize strictly for maximum overall Accuracy)
criterion = nn.CrossEntropyLoss()

train_losses, val_losses = [], []
train_accs, val_accs = [], []

print("Starting training process...")
for epoch in range(epochs):
    # Training phase
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
        
    train_loss = total_loss / len(train_loader)
    train_acc = 100 * correct / total
    
    # Validation phase
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            
    val_loss = val_loss / len(test_loader)
    val_acc = 100 * val_correct / val_total
    
    scheduler.step(val_loss)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

# Save final model weights
torch.save(model.state_dict(), 'optimized_resnet1d_100ep.pth')
print("Model weights saved to 'optimized_resnet1d_100ep.pth'")

# Export Ground Truth vs Prediction mapping
class_names = ['Ambient', 'Cargo', 'Passenger', 'Tanker', 'Tug']
predictions_df = pd.DataFrame({
    'Sample_Index': range(len(all_targets)),
    'Ground_Truth_Label': [class_names[t] for t in all_targets],
    'Predicted_Label': [class_names[p] for p in all_preds]
})
predictions_df.to_csv('predictions_vs_groundtruth.csv', index=False)
print("Saved 'predictions_vs_groundtruth.csv'")

# 3. Plotting & Reporting
sns.set_theme(style="whitegrid")

# Learning Curve Plot
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(val_losses, label='Validation Loss', linewidth=2, linestyle='--')
plt.title('Training & Validation Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy', linewidth=2)
plt.plot(val_accs, label='Validation Accuracy', linewidth=2, linestyle='--')
plt.title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('learning_curves_100ep.png', dpi=300)
print("Exported 'learning_curves_100ep.png'")

# Confusion Matrix Plot
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 12})
plt.title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
plt.ylabel('True Class', fontsize=12)
plt.xlabel('Predicted Class', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix_100ep.png', dpi=300)
print("Exported 'confusion_matrix_100ep.png'")

# Classification Report
report = classification_report(all_targets, all_preds, target_names=class_names, digits=4, zero_division=0)
with open("classification_report_100ep.txt", "w") as f:
    f.write(report)
print("Classification Report saved to 'classification_report_100ep.txt'.")
print("Process completed successfully!")
