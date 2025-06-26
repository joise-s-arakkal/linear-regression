# 🏠 Task 3: Linear Regression - House Price Prediction
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1grgYdx7MzAHSs1XVX4X746dBBf_RnHnq?usp=sharing)

## 📌 Objective

The goal of this task is to **implement and understand simple and multiple linear regression** models using the `Housing.csv` dataset. We aim to predict house prices based on multiple features such as area, number of bedrooms, location factors, etc.

---

## 🛠 Tools & Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 📂 Dataset

Dataset: [`Housing.csv`](Housing.csv)  
Source: [Kaggle - House Price Prediction Dataset](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction)

**Target Variable**: `price`  
**Features** include:
- `area`, `bedrooms`, `bathrooms`
- `stories`, `mainroad`, `guestroom`
- `basement`, `hotwaterheating`, `airconditioning`
- `parking`, `prefarea`, `furnishingstatus`

---

## 🧪 Steps Performed

1. **Data Preprocessing**  
   - Loaded CSV data using Pandas  
   - Converted categorical columns (`yes`/`no`) to binary (1/0)  
   - One-hot encoded `furnishingstatus`  
   - Removed the `price` column from features `X` and kept it as target `y`

2. **Train-Test Split**  
   - Used 80% for training and 20% for testing  
   - Random seed set for reproducibility

3. **Model Training**  
   - Fitted a `LinearRegression()` model from `sklearn.linear_model`

4. **Evaluation Metrics**  
   - MAE (Mean Absolute Error)  
   - MSE (Mean Squared Error)   
   - R² Score

5. **Visualization**  
   - Scatter plot of actual vs predicted prices  
   - Printed model coefficients for interpretation

---

## 📊 Results

| Metric     | Value      |
|------------|------------|
| MAE        | 970043.4039201637   |
| MSE        | 1754318687330.6633   |
| R² Score   | 0.6529242642153185      |

---

## 🔍 Key Learnings

- How to preprocess categorical variables for regression
- The significance of linear regression coefficients
- How evaluation metrics like R², MAE, and MSE indicate model performance
- Difference between **simple** and **multiple** regression models
- How to visualize model accuracy and interpret results
