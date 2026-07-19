# E-Commerce Customer Churn Prediction Project

## 📌 Project Overview
This project implements an end-to-end machine learning pipeline to predict customer churn for an e-commerce platform. By analyzing behavioral data and demographics, the goal is to identify high-risk customers and provide actionable insights for proactive retention strategies.

## 🛠️ Technical Stack
- **Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn
- **Methods:** Label Encoding, Feature Engineering, GridSearchCV, Cross-Validation

## 🚀 Project Workflow

### **Stage 1: Data Handling & Cleaning**
- Automated handling of missing values using median imputation for numerical data and mode for categorical data.
- Categorical feature encoding using `LabelEncoder` to transform text data into model-ready formats.

### **Stage 2: Advanced Feature Engineering**
Created high-impact domain-specific features to capture complex behaviors:
- **Support Intensity:** Customer service calls relative to membership length.
- **Abandonment Risk:** High LTV interaction with cart abandonment rates.
- **Efficiency Ratio:** Ratio of purchases to login frequency.

### **Stage 3: Model Training & Baseline Comparison**
- Benchmarked multiple algorithms including Logistic Regression, SVM, KNN, Random Forest, and XGBoost.
- Implemented `StandardScaler` for distance-sensitive models.

### **Stage 4: Hyperparameter Tuning**
- Conducted systematic optimization using `GridSearchCV` and `RandomizedSearchCV`.
- Optimized for **F1-Score** to ensure a balanced trade-off between Precision and Recall.

### **Stage 5: Final Evaluation & Champion Selection**
- **Champion Model:** Gradient Boosting Classifier
- **Final Accuracy:** 91.93%
- **F1-Score:** 0.8471
- **Key Drivers identified:** Customer Service Calls, Lifetime Value, and Cart Abandonment Rate.

### **Stage 6: Real-World Simulation**
- Validated the model using synthesized new customer data to ensure deployment readiness and consistent prediction outputs.

## 📊 Key Business Insights
1. **Friction Points:** High rates of customer service calls are the strongest predictor of churn, suggesting that resolving technical or service issues is critical for retention.
2. **Value at Risk:** The model specifically flags high-LTV customers who exhibit cart abandonment behavior, allowing the marketing team to offer targeted incentives to those with the highest revenue impact.
3. **Generalization:** A stable Cross-Validation score (91.51%) indicates the model is robust and not overfitted to historical data.

## 📝 How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the notebook to process data and generate the model.
3. Use the trained `final_gb_model` for real-time predictions.
