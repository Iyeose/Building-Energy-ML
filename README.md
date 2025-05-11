# Building-Energy-ML
# 🏠 Predicting Building Energy Consumption Using Machine Learning

This project investigates the use of machine learning models to predict **Heating Load (HL)** and **Cooling Load (CL)** in residential buildings. It compares multiple regression techniques and identifies key building design factors affecting energy consumption, contributing to sustainable architecture and efficient HVAC management.

## 📈 Project Overview
The building sector accounts for a large share of global energy consumption, with HVAC systems being major contributors. This study aims to leverage machine learning for accurate energy load forecasting, optimizing building performance, and reducing carbon emissions.

## 🎯 Objectives
- Predict residential **Heating Load** and **Cooling Load**.
- Compare the predictive performance of different ML algorithms.
- Identify key design features influencing energy consumption.
- Highlight the role of data preprocessing and feature selection.

## 🛠️ Methodology
- **Dataset**: Energy Efficiency dataset (UCI ML Repository).
- **Preprocessing**:
  - Exploratory Data Analysis (EDA)
  - Spearman and Partial Correlation Analysis
  - Z-score Standardization (avoiding data leakage)
  - Feature Selection (removing low-informative predictors)
- **Modeling**:
  - Decision Tree (DT)
  - Random Forest (RF)
  - Support Vector Machine (SVM)
  - Artificial Neural Network (ANN)
- **Evaluation Metrics**:
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R² (Coefficient of Determination)

## 🔍 Key Results
| Model         | Heating Load (HL) R² | Cooling Load (CL) R² |
|---------------|---------------------:|--------------------:|
| Decision Tree | 0.956                | 0.940               |
| Random Forest | 0.997                | 0.968               |
| SVM           | 0.994                | 0.952               |
| ANN           | **0.997**            | **0.969**           |

- **ANN and Random Forest** delivered superior predictive accuracy.
- **Glazing Area** emerged as the most influential feature.
- Ensemble methods (RF) balanced performance and interpretability.

## 📊 Visual Insights
- Predicted vs. Actual scatter plots show tight clustering for RF, SVM, and ANN.
- Random Forest feature importance highlighted Glazing Area, Surface Area, Wall Area, and Overall Height as key predictors.

## ⚙️ Files Included
- `ENB2012_data.csv`: Dataset used for model development.
- `Building_Energy_ml.R`: Complete R script with preprocessing and model code.
- `Final_Report.pdf`: Full academic report with visualizations, methodology, results, and references.

## 🚀 Future Work
- Incorporate advanced algorithms (XGBoost, LightGBM).
- Perform hyperparameter tuning (Grid Search, Bayesian Optimization).
- Implement model interpretability tools (SHAP, LIME).
- Validate models on real-world energy datasets.
- Deploy models as web applications for energy-efficient design tools.

## 📚 References
Key literature and datasets referenced are cited in the final report, including works by Tsanas & Xifara (2012), Zhao & Magoulès (2012), and more.

## 👤 Author
**Iyeose Simon Uhumuvabi**  
Advanced Decision Making: Predictive Analytics and Machine Learning

---

> This project is part of my academic portfolio, showcasing practical applications of machine learning in sustainable energy optimization.
