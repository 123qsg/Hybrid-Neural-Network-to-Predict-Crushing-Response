# Hybrid Neural Network for Predicting Crushing Response of Thin-Walled Tubular Structures

This repository contains the code for the paper:

**"Using Hybrid Neural Network to Predict Crushing Response for Millions of Thin-Walled Tubular Structures with Complex Configurations"**

The model predicts the axial crushing behavior of thin-walled tubes, including:

- **Energy absorption**
- **Maximum crushing force**
- **Full force-displacement curve** (200 points)

A two-stage hybrid architecture is used:
1. A **Multi-Layer Perceptron (MLP)** predicts energy and maximum force from structural features.
2. A **LSTM-based model augmented with attention** predicts the entire force-displacement curve using the same structural features and the outputs from the MLP.

Training uses K‑Fold cross‑validation with early stopping, and prediction ensembles all folds for robust inference.

---

## Code Structure

| File | Description |
|------|-------------|
| `MLP-LSTM-train.py` | Training script for both the target model and the curve model. |
| `MLP-LSTM-prediction.py` | Inference script that loads trained models and makes predictions on new data (numpy arrays or CSV files). |

---

## Data Files

- **Training data**: `3886simple.xlsx`  
  An Excel file (`.xlsx`) without a header row.  
  The first 24 columns are structural input features.  
  Columns 25 onward contain the 200-point force-displacement curve.
  
---

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Scikit‑learn
- SciPy
- tqdm
- joblib
- openpyxl

Install all dependencies with:

```bash
pip install tensorflow pandas numpy scikit-learn scipy tqdm joblib openpyxl
