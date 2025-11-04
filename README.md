<p align="center">
  <img src="docs/banner.png" width="600">
</p>

<h1 align="center">SoilML Toolbox</h1>
<p align="center"><b>Explainable Machine Learning Platform for Soil & Geotechnical Engineering</b></p>

<p align="center">
  <a><img src="https://img.shields.io/badge/python-3.8%2B-blue"></a>
  <a><img src="https://img.shields.io/badge/framework-XGBoost%20%2B%20SHAP-orange"></a>
  <a><img src="https://img.shields.io/badge/license-MIT-green"></a>
  <a><img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Mac-lightgrey"></a>
</p>

---

## 📘 Introduction

**SoilML Toolbox** is an open-source, GUI-based machine learning tool designed for soil and geotechnical engineering applications.

It provides a **transparent, reproducible, and practitioner-friendly workflow** for:

- Feature importance analysis (SHAP + XGBoost)
- Prediction of geotechnical parameters
- Engineering validation using 45° parity plots (R², RMSE, MAE)

📌 *No coding experience required — designed for both researchers and practitioners.*

---

## 🌟 Features

| Feature | Description |
|---|---|
✅ Load custom soil datasets (CSV) | Laboratory / field data compatible  
✅ One-click model training | XGBoost auto-pipeline  
✅ SHAP-NFI & XGBoost importance plots | Explainable ML  
✅ 45° parity plot | Classical geotechnical model validation  
✅ Export figures & tables | Publication-grade output  
✅ Broad applicability | SWCC, compression index (Cc), swelling %, k, UCS, etc.  

---

## 📂 Example Datasets

To support reproducibility and hands-on testing, two benchmark datasets are included in the `/data` directory.

### **1) Concrete Compressive Strength Dataset (UCI Repository)**

**Citation:**  
Yeh, I-C. (1998). *Modeling of strength of high-performance concrete using artificial neural networks.*  
Computers & Structures, 80(2–3), 131–141.  
📎 https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength

📁 File: `data/concrete_strength_UCI_1998.csv`

---

### **2) Soil Geotechnical Properties Dataset (Nigeria, Mendeley Data)**

**Citation:**  
Aderinola, O. S. et al. (2021). *Soil geotechnical properties dataset for a tremor area of South-West Nigeria.*  
Mendeley Data, V2.  
📎 https://doi.org/10.17632/cts2jkgvrw.2

Variables include: LL, PL, PI, Gs, water content, cohesion, φ, etc.

**Demonstrated here for:**  
Compression index *(Cc)* interpretation & prediction via explainable ML

📁 File: `data/nigeria_geotechnical_dataset.csv` *(processed subset)*

> The included CSV is a cleaned, standardized extract retaining original statistical behavior for immediate GUI use.

---

## 🛠️ Installation

### Option A — **Conda (recommended)**

```bash
git clone https://github.com/Higher777/SoilML-Toolbox.git
cd SoilML-Toolbox
conda env create -f environment.yml
conda activate soilml
python SoilML_Toolbox_GUI.py
