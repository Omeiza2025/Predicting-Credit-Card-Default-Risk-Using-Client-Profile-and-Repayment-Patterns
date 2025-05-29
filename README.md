# 🔤 Project Title: 
**Predicting Credit Card Default Risk Using Client Profile and Repayment Patterns**


## 📚 Table of Contents

1. [Project Title](#project-title)
2. [Introduction](#introduction)
3. [Objectives](#objectives)
4. [Technologies Used](#technologies-used)
5. [Dataset Source](#dataset-source)
6. [Expected Insights & Visualizations](#expected-insights--visualizations)
7. [Data Cleaning & Preprocessing](#data-cleaning--preprocessing)
8. [Feature Engineering & Modeling Preparation](#feature-engineering--modeling-preparation)
9. [Model Building & Evaluation](#model-building--evaluation)
10. [Confusion Matrix Analysis](#confusion-matrix-analysis)
11. [ROC Curve Comparison](#roc-curve-comparison)
12. [Feature Importance](#feature-importance)
13. [Conclusion](#project-conclusion)
14. [Key Takeaways](#key-takeaways)
15. [References](#references)


---

## 🧾 Project Report Structure

### 1. 📘 Introduction
This project focuses on understanding and predicting the likelihood that a credit card client will default on their payment in the next month. Using demographic, financial, and behavioral attributes, we explore factors that contribute to defaulting, helping financial institutions make better lending decisions.

---

### 2. 🎯 Objectives
- Identify key features influencing credit default risk  
- Build a predictive model to classify clients at risk of default  
- Provide actionable insights to reduce financial risk  
- Improve credit risk segmentation strategies  

---

### 3. 🛠 Technologies Used
- Python: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`  
- Jupyter Notebook for data exploration and model development  
- Power BI or Tableau (optional for dashboard visualization)  
- Microsoft Excel (for initial data management)  

---

### 4. 📂 Dataset Source
- Source: [UCI Machine Learning Repository – Default of Credit Card Clients](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)  
- Local File: `preprocessed_credit_default.csv`  

---

### 5. 🔍 Expected Insights
- Most influential features contributing to default  
- Patterns of default based on age, gender, education, or payment history  
- Payment behaviors or bill amounts that may signal credit risk  
- Profile segments with higher or lower default probabilities  

---

## 🔍 Expected Insights & Visualizations

This section outlines key questions explored through data visualization based on the cleaned credit card default dataset.

---

### 📊 Insight 1: Default vs. Non-Default Clients
 
What percentage of clients defaulted versus those who didn’t?

![insight_1_default_distribution](https://github.com/user-attachments/assets/9c5ec601-4d31-462a-b3f4-743bcc9f23a8)


**Analysis:**  
This chart illustrates the imbalance between defaulters and non-defaulters, meaning models must handle class imbalance carefully to avoid biased predictions.

---

### 🎓 Insight 2: Default Rate by Education Level


How does the default rate vary across different education levels?

![insight_2_default_by_education](https://github.com/user-attachments/assets/72752083-f5b8-4a4f-96fd-46cea2a69b6f)



**Analysis:**   
This bar chart shows that defaults occur across all education levels, suggesting that education alone does not strongly predict default risk but may help when combined with other features.

---

### 📈 Insight 3: Default Rate by Age Group



What is the relationship between age and default risk?


![insight_3_default_by_age](https://github.com/user-attachments/assets/db3da5f8-4602-4d34-80e2-e9f33e1dd6a9)

**Analysis:**  
This chart depicts that younger clients (ages 20–40) have higher default rates, indicating that age can be used for risk segmentation and targeting.

---

### 💳 Insight 4: Past Payment Statuses (PAY_1 to PAY_6)

How do past payment records relate to the likelihood of default?

![insight_payment_history_vs_default](https://github.com/user-attachments/assets/e88695e6-e666-4468-8cb9-58f8f073783a)



**Analysis:**  
This boxplot clearly shows defaulters have more delayed payments across all months, confirming that past payment behavior is a strong predictor of future default.


---

### 💰 Insight 5: Credit Limit vs. Default

 
Do clients with higher credit limits default more or less than others?

![insight_5_credit_limit_vs_default](https://github.com/user-attachments/assets/2b41095b-86d3-4996-a75f-21a5261ab6a1)


**Analysis:**   
This boxplot shows that defaulters generally have lower credit limits, implying that credit limit size may moderately relate to default risk.

---

### ❤️ Insight 6: Default Rate by Marital Status

Does marital status influence the likelihood of default?

![insight_6_default_by_marriage](https://github.com/user-attachments/assets/e61699b6-680b-4a13-a31b-6a0bc4869f88)



**Analysis:**   
This chart shows that single clients default slightly more than married ones, indicating that marital status has limited but possible predictive value.

---

## 🛠 Feature Engineering & Modeling Preparation

Before training the model, relevant features are selected and preprocessed. Categorical variables like `SEX`, `MARRIAGE`, and `EDUCATION` are encoded appropriately. Class imbalance is addressed using [chosen method], and numerical variables are scaled where necessary. A train-test split ensures proper model evaluation.

---

## Data Analysis 

## 🤖 Model Comparison & Evaluation
```python
evaluate_model("Logistic Regression", log_model, X_test, y_test)
evaluate_model("Decision Tree", tree_model, X_test, y_test)
evaluate_model("Random Forest", rf_model, X_test, y_test)

```

Three models were trained and evaluated to predict the likelihood of credit card default: **Logistic Regression**, **Decision Tree**, and **Random Forest**. Each model was assessed using four key metrics: accuracy, precision, recall, and F1-score — all crucial in handling the class imbalance in the dataset.

### 🔍 Evaluation Summary

| Model               | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression|  0.6795  |   0.3671  | 0.6202 |  0.4612  |
| Decision Tree      |  0.7307  |   0.3888  | 0.3806 |  0.3846  |
| Random Forest      |  0.8105  |   0.6342  | 0.3384 |  0.4413  |

### 💡 Interpretation:

- **Logistic Regression** achieved the **highest recall (0.62)**, making it the most sensitive in identifying defaulters. However, it has lower precision, which means more false positives.
- **Decision Tree** offered the most balanced precision and recall among interpretable models but underperformed in recall and F1-score.
- **Random Forest** delivered the **best accuracy (81%) and highest precision (0.63)**, meaning it confidently identifies defaulters with fewer false positives, though its recall is lower than logistic regression.

### ✅ Recommendation:
- Use **Logistic Regression** when prioritizing catching as many defaulters as possible (recall-sensitive use cases).
- Use **Random Forest** when it's more important to reduce false positives (e.g., in financial decision-making), and a balanced performance is preferred.


  ---

  ## ✅ Project Conclusion

This project aimed to predict the likelihood of credit card clients defaulting on their next payment using demographic, credit, and behavioral data. Through structured data cleaning, exploratory data analysis, feature engineering, and model comparison, we successfully developed and evaluated predictive models that can support decision-making in credit risk assessment.

Among the models tested, **Random Forest** achieved the highest overall accuracy and precision, making it the most reliable model for reducing false positives. **Logistic Regression**, on the other hand, showed superior recall, capturing more true defaulters, which is critical when the cost of missing a defaulter is high. **Decision Tree** served as a balanced, interpretable model with moderate performance.

By understanding the drivers of default, such as recent payment delays, credit limit, and bill/re-payment patterns, financial institutions can make more informed, data-driven decisions regarding customer risk profiling.

---

## 📌 Key Takeaways

- 🔍 **Payment history variables (`PAY_0` to `PAY_6`)** are the most influential predictors of default, indicating that recent repayment behavior is a strong signal of financial risk.

- ⚖️ The dataset is **highly imbalanced**, with defaulters making up only ~22%. This required careful model evaluation using precision, recall, and F1-score rather than accuracy alone.

- 🤖 **Random Forest** provided the most balanced and reliable performance, especially for minimizing false positives — suitable for conservative credit approval processes.

- 📈 **Logistic Regression** was best for identifying the most defaulters (high recall), making it ideal for risk mitigation strategies such as customer follow-up or intervention.

- 📊 **Exploratory data analysis** revealed that age, education, marital status, and credit limit all contribute to default risk but are weaker predictors compared to behavioral indicators.

---



## 📚 References

1. **UCI Machine Learning Repository – Default of Credit Card Clients Dataset**  
   https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

2. **Scikit-learn Documentation** – Machine Learning in Python  
   https://scikit-learn.org/stable/documentation.html

3. **Pandas Documentation** – Data Analysis in Python  
   https://pandas.pydata.org/docs/

4. **Seaborn Documentation** – Statistical Data Visualization  
   https://seaborn.pydata.org/

5. **Matplotlib Documentation** – Plotting Library for Python  
   https://matplotlib.org/stable/index.html

6. **StandardScaler – scikit-learn Preprocessing**  
   https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

7. **ROC and AUC explained** – Towards Data Science  
   https://towardsdatascience.com/understanding-roc-and-auc-436cf8c3e003





