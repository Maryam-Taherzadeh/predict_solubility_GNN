# Solubility Prediction App

# 🧪 Predict Solubility with Graph Neural Network

This app predicts the water solubility (logS) of small molecules using a Graph Neural Network (GNN) trained on the AqSolDB dataset.

## 🔬 Model Details
- **Architecture**: GCN with atom and molecular-level features
- **Training Dataset**: AqSolDB (over 9,800 molecules)
- **Input**: Molecule name or SMILES
- **Output**: Predicted logS (solubility)

## 🚀 How to Use
1. Enter a molecule name (e.g., "Caffeine", "Ibuprofen")
2. View the molecular structure
3. Get the predicted water solubility (logS)

## 📦 Requirements
- RDKit
- PyTorch
- PyTorch Geometric
- Streamlit

## 🤖 Powered By
Graph Neural Networks + RDKit Descriptors + PyTorch Geometric

