
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# ---- Step 1: Generate and Normalize Data ----
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                           n_clusters_per_class=2, random_state=42)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Standardize data for better distance calculations
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Reduce noise before UMAP
pca = PCA(n_components=15)  # Reduce feature noise
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# Convert to Torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)


# ---- Step 2: Define Improved UMAP-like Embedding Model ----
class UMAPNet(nn.Module):
    def __init__(self, input_dim, n_components=10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_components)
        )
    
    def forward(self, X):
        return self.fc(X)


# ---- Step 3: Train UMAP Embedding Model with Triplet Loss ----
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=margin)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)

def train_umap_model(model, X_train, n_neighbors=30, epochs=500, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = TripletLoss(margin=0.5)  # Triplet loss for better separation

    for epoch in range(epochs):
        optimizer.zero_grad()
        embeddings = model(X_train)

        # Compute pairwise distances
        D = torch.cdist(embeddings, embeddings, p=2)
        knn_indices = torch.argsort(D, dim=1)[:, 1:n_neighbors+1]

        # Select triplets
        anchor = embeddings
        positive = embeddings[knn_indices[:, 0]]  # First neighbor
        negative = embeddings[knn_indices[:, -1]]  # Farthest neighbor

        loss = criterion(anchor, positive, negative)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")

    return model

umap_model = UMAPNet(input_dim=X_train.shape[1], n_components=10)
umap_model = train_umap_model(umap_model, X_train)


# Transform data
X_train_umap = umap_model(X_train).detach().numpy()
X_test_umap = umap_model(X_test).detach().numpy()

# ---- Step 4: Train an XGBoost Classifier ----
clf = XGBClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
clf.fit(X_train_umap, y_train)
y_pred = clf.predict(X_test_umap)

# ---- Step 5: Evaluate Performance ----
accuracy = accuracy_score(y_test, y_pred)
print(f'🔹 Classification Accuracy: {accuracy:.4f}')

# ---- Step 6: Visualization ----
plt.scatter(X_train_umap[:, 0], X_train_umap[:, 1], c=y_train, cmap='coolwarm', alpha=0.5, label='Train')
plt.scatter(X_test_umap[:, 0], X_test_umap[:, 1], c=y_test, cmap='coolwarm', marker='x', label='Test')
plt.legend()
plt.title("UMAP Projection of Data (Torch)")
plt.show()

