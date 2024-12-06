# HeartAttackAnalysis-Prediction


# Heart Attack Analysis and Prediction

## **Project Overview**
This project focuses on analyzing and predicting heart attack risks using machine learning models. The dataset, sourced from Kaggle, contains key medical attributes related to heart health. By preprocessing, visualizing, and modeling the data, we aim to identify patterns and build predictive models for heart attack outcomes.

## **Dataset**
- **Heart Attack Analysis Dataset**: [Kaggle Link](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset?resource=download&select=heart.csv)

## **Steps in the Project**

### **1. Data Preprocessing**
- Renamed columns for clarity, e.g., `cp` → `Chest Pain Type`, `chol` → `Cholesterol`.
- Handled missing values by dropping rows with nulls.
- Provided concise summaries and descriptive statistics for better understanding.

### **2. Data Visualization**
- Used count plots to explore categorical features like sex, chest pain type, and fasting blood sugar.
- Visualized correlations with a heatmap.
- Generated pair plots for feature relationships.

### **3. Modeling**
- Target Variable: `target variable` (0: No heart attack, 1: Heart attack).
- Feature Set: Remaining attributes after preprocessing.
- Splitting Dataset: 80% training, 20% testing.
- Implemented models:
  - Decision Tree Classifier
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest Classifier
  - K-Nearest Neighbors (KNN)
  - Gradient Boosting Classifier
- Evaluated models based on accuracy and confusion matrices.

### **4. Feature Importance Analysis**
- Used Decision Tree Classifier to determine feature importance.
- Visualized feature importance with bar plots.

### **5. Cross-Validation**
- Performed 10-fold cross-validation to ensure robustness.
- Analyzed feature importance across folds.

## **Dependencies**
- **Python Libraries**:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn

## **How to Run**
1. Download the dataset and place it in the project directory.
2. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
3. Run the Python script to preprocess data, visualize trends, and build predictive models.
4. Check output files and model evaluation metrics for insights.

## **Key Insights**
- Heatmaps and pair plots reveal relationships between features.
- Decision Tree visualization
