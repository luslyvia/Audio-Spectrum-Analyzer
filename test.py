import os
import sys
import subprocess
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score

REPO_NAME = "resnet1d"

if not os.path.exists(REPO_NAME):
    subprocess.run(["git", "clone", "https://github.com/hsd1503/resnet1d.git"], check=True)

current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, REPO_NAME))

try:
    from resnet1d import ResNet1D_Binary
except ImportError:
    from models import ResNet1D_Binary 

model = ResNet1D_Binary()

checkpoint_path = 'optimized_resnet1d_100ep.pth'
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Không tìm thấy file {checkpoint_path}.")

pretrained_dict = torch.load(checkpoint_path, map_location='cpu')
model_dict = model.state_dict()

filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'dense' not in k}
model_dict.update(filtered_dict)
model.load_state_dict(model_dict)

data_path = 'test_set.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Không tìm thấy file {data_path}.")

df = pd.read_csv(data_path)
X_raw = df.iloc[:, :-1].values.astype(np.float32)
y_true = df.iloc[:, -1].apply(lambda x: 1 if x > 0 else 0).values

X_norm = (X_raw - X_raw.mean(axis=0)) / (X_raw.std(axis=0) + 1e-6)
X_tensor = torch.tensor(X_norm).view(-1, 1, X_raw.shape[1])
y_tensor = torch.tensor(y_true, dtype=torch.float32)

optimizer = torch.optim.Adam(model.dense.parameters(), lr=0.001)
criterion = nn.BCELoss()

model.train()
for epoch in range(20):
    optimizer.zero_grad()
    outputs = model(X_tensor).squeeze()
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/20 - Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    probs = model(X_tensor).squeeze().numpy()
    optimal_threshold = np.mean(probs)
    y_pred = (probs > optimal_threshold).astype(int)
    print(f"\nNgưỡng tối ưu được chọn: {optimal_threshold:.4f}\n")

accuracy = accuracy_score(y_true, y_pred)

print("--- PERFORMANCE REPORT ---")
print(f"Overall Accuracy: {accuracy*100:.2f}%")
print(classification_report(y_true, y_pred, target_names=['No Vessel', 'Vessel']))

cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(6, 5))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=['No Vessel', 'Vessel'],
            yticklabels=['No Vessel', 'Vessel'])
plt.title('Normalized Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

fpr, tpr, _ = roc_curve(y_true, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='blue')
plt.plot([0, 1], [0, 1], 'r--', color='red')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

report_dict = classification_report(y_true, y_pred, output_dict=True)
df_plot = pd.DataFrame(report_dict).transpose().iloc[:2, :-1]
df_plot.plot(kind='bar', figsize=(7, 4), rot=0)
plt.title(f'Performance Metrics Summary (Acc: {accuracy*100:.2f}%)')
plt.ylim(0, 1.1)
plt.ylabel('Score')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()