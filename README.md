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

It provides a **transparent, reproducible, and practitioner-oriented workflow** for:

- Feature significance interpretation (SHAP + XGBoost)
- Prediction of soil engineering parameters
- Model accuracy visualization (45° parity plots)

📌 *No coding experience required — designed for researchers & practitioners.*

---

## 🌟 Features

| Feature | Description |
|---|---|
✅ Load custom soil datasets (CSV) | Import your own laboratory / site data  
✅ Auto-train XGBoost model | One-click machine learning  
✅ SHAP-NFI + XGBoost importance | Transparent feature influence ranking  
✅ 45° parity plot (R², RMSE, MAE) | Classical civil/geo ML validation  
✅ Export plots & tables | Publication-ready outputs  
✅ General-purpose | SWCC, Cc, swelling index, permeability, UCS, etc.  

---

## 🛠️ Requirements & Install

### Option A — **Conda (recommended)**

```bash
git clone https://github.com/<your-repo>/SoilML-Toolbox.git
cd SoilML-Toolbox
conda env create -f environment.yml
conda activate soilml
python SoilML_Toolbox_GUI.py
