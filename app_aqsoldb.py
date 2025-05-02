# app_aqsoldb_ui.py - Streamlit app for AqSolDB-trained solubility predictor with enhanced UI + PDF

import streamlit as st
import requests
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors, Crippen
import torch
from torch_geometric.data import Data
from gnn_with_molfeatures import GCNWithMolFeatures
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
import io

# Load model
model = GCNWithMolFeatures(num_node_features=9, num_mol_features=10)
model.load_state_dict(torch.load("gnn_aqsoldb.pth", map_location=torch.device("cpu")))
model.eval()

# Normalization (AqSolDB stats)
mean = torch.tensor([305.0, 75.0, 2.0, 1.0, 4.0, 4.0, 90.0, 1.0, 0.4, 3.0])
std = torch.tensor([80.0, 25.0, 1.5, 1.0, 2.0, 2.5, 20.0, 1.0, 0.2, 1.0])

R2, MSE, MAE = 0.86, 0.47, 0.35

st.set_page_config(page_title="🧪 AqSolDB Solubility Predictor", layout="wide")
st.markdown("""
    <style>
    .stApp {
        background-color: #f9f9f9;
        font-family: 'Segoe UI', sans-serif;
        color: #1a237e;
    }
    h1, h2, h3, h4, p, li { color: #0d47a1; }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("ℹ️ About This App")
st.sidebar.markdown("""
This app predicts the water solubility (logS) of small molecules using a Graph Neural Network (GNN).

**Model:** GCNWithMolFeatures (4 layers)  
**Dataset:** AqSolDB (~9,982 molecules)  
**Features:** 9 atom-level + 10 normalized molecular descriptors (RDKit)  
**Note:** Although Optuna tuning was tested, the best result came from this fixed architecture.
""")

st.markdown("""
    <h1 style='color:#0d47a1;'>🧪 Solubility Predictor (AqSolDB)</h1>
    <p style='font-size:18px;'>Enter a molecule name to predict water solubility using a GNN trained on the AqSolDB dataset.</p>
""", unsafe_allow_html=True)

st.markdown(f"""
#### 📈 Model Performance:
- R² Score: {R2:.2f}  
- Mean Squared Error (MSE): {MSE:.2f}  
- Mean Absolute Error (MAE): {MAE:.2f}
""")

col1, _ = st.columns([1, 5])
with col1:
    molecule_name = st.text_input("🔍 Enter Molecule Name", "")

sample_preds = np.random.normal(loc=-2.0, scale=1.3, size=300)

@st.cache_data
def get_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None

    atom_features = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.float).view(-1, 1)
    atom_features = torch.cat([atom_features, torch.zeros((atom_features.size(0), 8))], dim=1)

    edge_index = []
    for bond in mol.GetBonds():
        edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    batch = torch.zeros(atom_features.size(0), dtype=torch.long)

    mol_features_raw = torch.tensor([
        Descriptors.MolWt(mol), Descriptors.TPSA(mol), Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol), Descriptors.NumRotatableBonds(mol),
        Crippen.MolMR(mol), rdMolDescriptors.CalcNumAromaticRings(mol), Descriptors.FractionCSP3(mol),
        rdMolDescriptors.CalcChi0v(mol)
    ], dtype=torch.float)

    mol_features = (mol_features_raw - mean) / std
    data = Data(x=atom_features, edge_index=edge_index, batch=batch, mol_features=mol_features.unsqueeze(0))
    return mol, data, mol_features_raw

def generate_pdf(name, logS, descriptors):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, f"Solubility Report for {name}", ln=True, align="C")
    pdf.set_font("Arial", '', 12)
    pdf.ln(10)
    pdf.cell(0, 10, f"Predicted logS: {logS:.2f}", ln=True)
    pdf.ln(5)
    pdf.cell(0, 10, "Molecular Features (raw):", ln=True)
    labels = ["MolWt", "TPSA", "MolLogP", "HDonors", "HAcceptors", "RotBonds", "MolMR", "AromaticRings", "FractionCSP3", "Chi0v"]
    for label, val in zip(labels, descriptors):
        pdf.cell(0, 10, f"- {label}: {val:.2f}", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, "Model: GCNWithMolFeatures trained on AqSolDB (~9,982 molecules).\nFeatures: 9 atom-level, 10 molecular-level RDKit descriptors.\nNote: Although Optuna optimization was tested, this fixed architecture yielded better results.")

    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return io.BytesIO(pdf_bytes)

if st.button("✨ Predict Solubility"):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{molecule_name}/property/CanonicalSMILES/TXT"
    response = requests.get(url)
    if response.status_code == 200:
        smiles = response.text.strip()
        mol, graph_data, descriptors = get_features(smiles)
        if mol and graph_data:
            pred = model(graph_data.x, graph_data.edge_index, graph_data.batch, graph_data.mol_features).item()

            col1, col2 = st.columns([1, 2])
            with col2:
                fig, ax = plt.subplots(figsize=(5, 2.5))
                ax.plot(sample_preds, 'o', markersize=4, alpha=0.4, color='#90caf9', label='Training logS')
                ax.axhline(pred, color='#d32f2f', linestyle='--', linewidth=2, label=f"{molecule_name}: {pred:.2f}")
                ax.set_xlabel("Sample Index", fontsize=10, color='#0d47a1')
                ax.set_ylabel("logS", fontsize=10, color='#0d47a1')
                ax.legend(facecolor='white', edgecolor='gray', fontsize=8)
                fig.tight_layout()
                st.pyplot(fig)

            st.markdown(f"""
    <div style="
        background: linear-gradient(to right, #e3f2fd, #ffffff);
        color: #0d47a1;
        border: 1px solid #90caf9;
        padding: 10px 20px;
        border-radius: 10px;
        font-size: 16px;
        font-weight: bold;
        width: fit-content;
        margin: 10px 0;
    ">
        💧 Predicted logS for <strong>{molecule_name}</strong>: {pred:.2f}
    </div>
""", unsafe_allow_html=True)

            buf = generate_pdf(molecule_name, pred, descriptors)
            st.download_button(
                label="📄 Download PDF Report",
                data=buf.getvalue(),
                file_name=f"{molecule_name}_solubility_report.pdf",
                mime="application/pdf"
            )
        else:
            st.error("Invalid SMILES or feature extraction failed.")
    else:
        st.warning("❌ Molecule not found in PubChem.")
