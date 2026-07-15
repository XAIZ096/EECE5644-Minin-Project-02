Project Overview
This project uses the Medical Cost Personal Dataset to help HealthGuard Insurance understand the factors that affect medical charges, predict annual customer costs, and identify customers whose charges are likely to be above the median.

Dataset
Download the dataset from https://www.kaggle.com/datasets/mirichoi0218/insurance, rename the downloaded file to insurance.csv, and place it in the same folder as the notebook. The dataset contains age, sex, BMI, number of children, smoking status, region, and annual medical charges.

Model Development & Evaluation
Requirement 2: Predicting Medical Charges (Regression)
Built and compared 7 regression model types to predict charges:

Simple Linear Regression (using smoker_yes, the top single predictor from EDA)
Multiple Linear Regression (all features)
Polynomial Regression (degrees 2–4)
Ridge Regression (L2, tuned α)
Lasso Regression (L1, tuned α)
Support Vector Regression (RBF and linear kernels)
Decision Tree Regression (tuned max_depth)

Setup: 80/20 train/test split (random_state=42), features standardized for SVR (target also scaled for SVR to avoid default epsilon issues).
Best model: Decision Tree Regression (max_depth=4) — R²=0.897, MAE=2621.31, RMSE=4345.88. Recommended for its top accuracy and interpretability (splits directly on smoker_yes, age, bmi, matching EDA insights).
Full comparison table and per-model discussion (including overfitting analysis at higher polynomial degrees / tree depths) is in the technical report.

Requirement 3: Flagging Expensive Customers (Classification)
Defined "expensive" as charges above the dataset median ($9,386.16), producing a balanced binary target (669 vs. 668).
Trained and compared:

Logistic Regression — Accuracy 0.907, Precision 0.898, Recall 0.918, F1 0.908
Decision Tree Classifier (max_depth=5) — Accuracy 0.940, Precision 0.992, Recall 0.888, F1 0.937

Best model: Decision Tree Classifier (max_depth=5) — higher accuracy and precision, with the tradeoff of slightly lower recall than Logistic Regression noted in the report.

How to Reproduce
Load insurance_cleaned.csv (produced by the data cleaning notebook)
One-hot encode categorical columns (sex, smoker, region)
Run the training notebook cells in order — model definitions, tuning sweeps, and final evaluation are laid out sequentially, with results collected in a results dictionary and displayed as a summary table

How to Run
Install the libraries listed in requirements.txt, open the notebook in Jupyter Notebook or JupyterLab, and run all cells from top to bottom. The cleaning section creates insurance_cleaned.csv, which is then used for the regression and classification models.
