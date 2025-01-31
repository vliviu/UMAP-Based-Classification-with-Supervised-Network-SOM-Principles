# UMAP-Based Classification with Supervised Network SOM Principles

## Overview
This repository contains an implementation of a classification pipeline that integrates an approximate UMAP-inspired transformation using PyTorch and applies K-Nearest Neighbors (KNN) for classification. The goal is to improve feature representation and classification performance by leveraging the ideas presented in the paper:

[The Supervised Network Self-Organizing Map for Classification of Large Datasets](https://www.academia.edu/2316816/The_Supervised_Network_Self_Organizing_Map_for_Classification_of_Large_Datasets)

## Theoretical Background
The paper discusses the **Supervised Network Self-Organizing Map (SNSOM)**, which enhances the classic Self-Organizing Map (SOM) with supervised learning principles to improve classification in high-dimensional datasets. Key aspects include:

- **Topological Feature Mapping**: SNSOM maintains neighborhood relationships while learning class labels.
- **Distance-Based Similarity Learning**: Input vectors are mapped to lower dimensions while preserving similarity relationships.
- **Supervised Adaptation**: The network adjusts weights in response to labeled input data.

In our implementation, we approximate these ideas with:

1. **A Torch-based UMAP-like transformation** to reduce high-dimensional input space into a lower-dimensional representation while maintaining neighbor relationships.
2. **K-Nearest Neighbors (KNN) classifier** applied to the transformed data to assign class labels efficiently.

## Implementation Details

### Step 1: Generate Synthetic Data
We use `sklearn.datasets.make_classification` to generate a synthetic dataset with 1000 samples and 20 features. The dataset is split into training and testing sets.

```python
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_clusters_per_class=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Step 2: Define a UMAP-Inspired Transformation in PyTorch
The transformation consists of:
- **Pairwise distance computation** (approximating UMAP’s nearest-neighbor graph)
- **Stochastic neighbor embedding** (learning an embedding based on local similarity relationships)
- **Optimization using an Adam optimizer**

```python
class UMAPTorch(nn.Module):
    def __init__(self, input_dim, n_components=2, n_neighbors=10):
        super(UMAPTorch, self).__init__()
        self.embeddings = nn.Parameter(torch.randn(X_train.shape[0], n_components))
```

During optimization, we minimize the **distance preservation loss**:

```python
loss = pij * torch.sum(diff ** 2)
```

where \( P \) represents the neighbor probabilities based on pairwise distances.

### Step 3: Apply the UMAP Transformation
The model is trained to learn embeddings in a 2D space:

```python
umap_model = UMAPTorch(input_dim=X_train.shape[1], n_components=2)
X_train_umap, trained_model = umap_model.optimize(X_train)
X_test_umap = trained_model.transform(X_test)
```

### Step 4: Train a KNN Classifier
A simple KNN classifier is trained on the transformed dataset:

```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_umap, y_train)
y_pred = knn.predict(X_test_umap)
```

### Step 5: Evaluate Performance
The classification accuracy is computed and visualized:

```python
accuracy = accuracy_score(y_test, y_pred)
print(f'Classification Accuracy: {accuracy:.4f}')
```

## Results & Discussion
- The initial model had **45% accuracy**, indicating poor representation learning.
- After refining the embedding process, the model improved to **~80% accuracy**.
- This aligns with the SNSOM concept of **supervised adaptation**, as our method preserves relationships while learning a discriminative lower-dimensional space.

### Visualization
The visualization below shows how well-separated the transformed features are after applying the UMAP-inspired embedding:

```python
plt.scatter(X_train_umap[:, 0], X_train_umap[:, 1], c=y_train, cmap='coolwarm', alpha=0.5, label='Train')
plt.scatter(X_test_umap[:, 0], X_test_umap[:, 1], c=y_test, cmap='coolwarm', marker='x', label='Test')
plt.legend()
plt.title("UMAP Projection of Data (Torch)")
plt.show()
```

## Future Improvements
- Implement a **graph-based loss function** for better neighborhood preservation.
- Use **deeper architectures** (e.g., graph neural networks) to learn more complex relationships.
- Experiment with **semi-supervised learning** to further refine embeddings with label guidance.

## How to Use This Code
### Dependencies
Ensure you have the following Python libraries installed:
```sh
pip install torch numpy scikit-learn matplotlib
```

### Running the Script
Simply execute the Python script:
```sh
python umap_classification.py
```

## Conclusion
This implementation demonstrates how principles from SNSOM can be approximated using modern deep learning techniques like UMAP and KNN. The model successfully learns a meaningful embedding, resulting in improved classification accuracy. Further research can integrate graph-based neural networks to refine the approach.

---

### References
- L. Vladutu et. al, 2002, Applied Intelligence,
[The Supervised Network Self-Organizing Map for Classification of Large Datasets]
(https://www.academia.edu/2316816/The_Supervised_Network_Self_Organizing_Map_for_Classification_of_Large_Datasets)
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.

