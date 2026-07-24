# Predicting Mental Health Treatment-Seeking Behavior Among Tech Employees Using Machine Learning
Analyzed the OSMI Mental Health in Tech Survey using Python, performed exploratory data analysis and feature selection, and built machine learning models to predict mental health treatment-seeking behavior. The best-performing model, Random Forest, achieved 81.67% accuracy and an ROC-AUC score of 91.49%.

## Project Overview

Mental health is a growing concern in the technology industry, where workplace pressure, long working hours, and stigma surrounding mental health can influence whether employees seek professional support. Although awareness of mental health has improved, many employees still face barriers to accessing treatment, making it important to understand the factors that drive treatment-seeking behavior.

This project analyzes the Mental Health in Tech Survey using an end-to-end data science workflow. It begins with data cleaning and exploratory data analysis (EDA) in Python to uncover patterns and relationships within the data. Feature selection techniques—including Chi-Square tests, Mutual Information, and Random Forest Feature Importance—are then used to identify the variables most strongly associated with treatment-seeking behavior. Finally, multiple machine learning models are developed and compared to predict whether an employee is likely to seek treatment for a mental health condition.

Rather than focusing solely on prediction, this project demonstrates a complete machine learning workflow—from data preprocessing and exploratory analysis to feature engineering, model development, evaluation, and interpretation. The findings provide valuable insights into the personal and workplace factors that influence mental health treatment-seeking behavior, offering practical recommendations for organizations seeking to improve employee mental well-being.

## Project Objectives
This project was designed to achieve four primary objectives:

- Analyze mental health treatment-seeking behavior among employees in the technology industry.
- Identify the personal and workplace factors most strongly associated with mental health treatment-seeking behavior.
- Develop and compare multiple machine learning classification models capable of predicting whether an employee is likely to seek treatment for a mental health condition.
- Evaluate model performance and generate insights that can support evidence-based workplace mental health initiatives.

## Tools & Technologies Used

### Programming Language

- **Python** – Data preprocessing, analysis, visualization, and machine learning.

### Development Environment

- **Jupyter Notebook** – Interactive environment for data analysis, experimentation, and model development.

### Python Libraries

#### Data Manipulation

| Library | Purpose |
|---------|---------|
| Pandas | Data cleaning, preprocessing, and transformation |
| NumPy | Numerical computations and array operations |

#### Data Visualization

| Library | Purpose |
|---------|---------|
| Matplotlib | Statistical and exploratory visualizations |
| Seaborn | Advanced data visualization and distribution plots |

#### Machine Learning

| Library/Technique | Purpose |
|-------------------|---------|
| Scikit-learn | Model development, preprocessing, feature selection, and evaluation |
| Logistic Regression | Classification model for predicting mental health treatment outcomes |
| Decision Tree Classifier | Tree-based classification model |
| Random Forest Classifier | Ensemble model and feature importance analysis |
| Train-Test Split | Dividing data into training and testing sets |
| Label Encoding | Converting categorical variables into numerical format |
| One-Hot Encoding | Creating binary variables for categorical features |
| Chi-Square Feature Selection | Selecting statistically significant categorical features |
| Mutual Information | Measuring predictive relationships between features and target |
| Performance Metrics | Accuracy, Precision, Recall, F1-Score, and ROC-AUC evaluation |

### Statistical Analysis

| Technique | Purpose |
|-----------|---------|
| Chi-Square Test | Evaluated associations between categorical variables and the target variable |
| Mutual Information | Measured the predictive relevance of each feature |
| Random Forest Feature Importance | Ranked variables based on their contribution to model predictions |

## Data Cleaning & Preprocessing Summary

i Removed irrelevant features

- Dropped columns that did not contribute to predicting treatment-seeking behavior.
- Removed unnecessary survey metadata and high-noise features where applicable.

ii Handled missing values

- Identified columns with missing responses.
- Removed features with excessive missing values.
- Preserved important variables where missingness could still provide useful information.

iii Corrected data types

- Converted columns into appropriate formats for analysis and machine learning.
- Ensured numerical and categorical variables were correctly identified.

iv Cleaned age-related inconsistencies

- Detected unrealistic age values and invalid responses.
- Removed values outside the expected working-age range.
- Verified the final age distribution for accuracy.

v Standardized categorical variables

- Cleaned inconsistent responses in categorical fields (e.g., gender variations).
- Grouped similar responses to improve data consistency.

vi Encoded categorical features

- Converted categorical variables into numerical representations.
- Applied encoding techniques such as One-Hot Encoding and Label Encoding for machine learning compatibility.

vi Prepared data for modeling

- Split the processed dataset into training and testing sets.
- Ensured the final dataset was structured for exploratory analysis and machine learning model development.

## Exploratory Data Analysis (Python)

Before building the machine learning models, an Exploratory Data Analysis (EDA) was conducted to understand the characteristics of the survey respondents, identify patterns within the data, and explore factors associated with mental health treatment-seeking behavior.

- What is the age distribution of survey respondents?
- What is the gender distribution of respondents?
- Which countries contributed the highest number of survey responses?
- How are respondents distributed based on whether they have sought treatment for a mental health condition?
- Is there an association between family history of mental illness and treatment-seeking behavior?
- Does the extent to which mental health interferes with work relate to treatment-seeking behavior?
- Does company size influence employees' likelihood of seeking treatment?
- Does the ease of obtaining medical leave for a mental health condition relate to treatment-seeking behavior?

## Key EDA Findings

The exploratory analysis revealed several notable patterns:

- Most respondents were between 27 and 36 years of age, with a median age of 31 years.
- The majority of respondents identified as male, reflecting the demographic composition of the survey.
- The United States accounted for the highest number of survey responses, followed by several other countries with substantially fewer participants.
- The target variable (Have you sought treatment for a mental health condition?) showed a relatively balanced distribution between respondents who had sought treatment and those who had not.
- Respondents with a family history of mental illness were more frequently observed among those who reported seeking treatment.
- Employees who reported that their mental health interfered with their work were more commonly found among respondents who had sought treatment.
- Treatment-seeking behavior varied across company sizes, suggesting that organizational characteristics may influence employees' willingness or ability to access mental health care.
- Employees who reported that it was easier to take medical leave for a mental health condition also showed different treatment-seeking patterns, highlighting the potential importance of workplace support policies.

Absolutely. Below is the equivalent version tailored to **your mental health treatment prediction project**.

## Machine Learning

While the exploratory analysis provided insights into the characteristics of the respondents and workplace factors, machine learning was used to answer the following question:

> Can we predict whether a technology employee is likely to seek treatment for a mental health condition based on demographic, personal, and workplace characteristics?

This project frames the problem as a Supervised Binary Classification task.

## Target Variable

Have you sought treatment for a mental health condition?

Target Classes:

Yes
No

## Feature Selection

Following the exploratory analysis, multiple feature selection techniques were applied to identify the variables most relevant for prediction.

The selected features included demographic, personal, and workplace-related variables such as:

- Age
- Gender
- Country
-  Family history of mental illness
-  Mental health interference with work
-  Company size
-  Employer mental health benefits
-  Availability of mental health resources
-  Ease of taking medical leave
-  Workplace attitudes toward mental health

Feature relevance was evaluated using:
- Chi-Square Test
- Mutual Information
- Random Forest Feature Importance

These methods helped identify the variables that contributed most to the prediction task while reducing unnecessary complexity.

## Data Preprocessing

Before training the models:
- Categorical variables were encoded using **One-Hot Encoding**.
- The target variable was encoded into binary classes.
- The dataset was divided into **80% training** and **20% testing** sets.
- A fixed **random_state** was used to ensure reproducible results.

## Models Developed

Three supervised classification algorithms were trained and evaluated.

## Logistic Regression

Used as the baseline classification model to establish benchmark performance.

## Decision Tree Classifier

Used to capture non-linear relationships between employee characteristics and treatment-seeking behavior.

## Random Forest Classifier

An ensemble learning algorithm consisting of **200 decision trees**, designed to improve prediction accuracy while reducing overfitting.

## Model Evaluation
Each model was evaluated using the following classification metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score

These metrics provide a comprehensive assessment of model performance, balancing overall accuracy with the model's ability to correctly identify employees who sought treatment.

# Results

| Model               |   Accuracy |  Precision |     Recall |   F1-Score |
| ------------------- | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     69.72% |     69.47% |     71.65% |     70.54% |
| Decision Tree       |     77.29% |     78.23% |     76.38% |     77.29% |
| **Random Forest**   | **81.67%** | **79.56%** | **85.83%** | **82.58%** |

**ROC-AUC (Random Forest): 91.49%**

## Best Performing Model

The **Random Forest Classifier** achieved the strongest overall performance.

It achieved:

* **Accuracy:** 81.67%
* **Precision:** 79.56%
* **Recall:** 85.83%
* **F1-Score:** 82.58%
* **ROC-AUC:** 91.49%

These results indicate that the Random Forest model was the most effective at distinguishing between respondents who had sought treatment and those who had not.

## Key Insights
The machine learning analysis revealed several important findings:
- Mental health interference with work was the strongest predictor used by the model.
- Family history of mental illness consistently ranked among the most influential predictors.
- Workplace-related factors—including employer mental health benefits, awareness of available resources, company size, and ease of taking medical leave—contributed meaningfully to the model's predictions.
- Demographic characteristics such as age and country also influenced prediction performance.
- The Random Forest model substantially outperformed Logistic Regression and Decision Tree, demonstrating the value of ensemble learning for this classification problem.
  
## Limitations
Although the models performed well, several limitations should be considered.
- The analysis is based on self-reported survey responses, which may introduce response bias.
- The dataset represents respondents from the 2014 OSMI Mental Health in Tech Survey and may not fully represent today's technology workforce.
- The model identifies predictive patterns rather than causal relationships.
- Some important factors influencing treatment-seeking behavior, such as personal financial circumstances, healthcare accessibility, and organizational culture, were not directly measured.
- Model evaluation was performed using a single train-test split. Cross-validation would provide a more robust estimate of model performance.

## Future Improvements
Potential enhancements include:
- Implementing K-Fold Cross-Validation.
- Performing hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
- Evaluating additional classification algorithms such as XGBoost, LightGBM, and CatBoost.
- Applying SHAP values to improve model interpretability.
- Combining multiple years of the OSMI Mental Health Survey to improve model generalization.
- Deploying the best-performing model as an interactive web application using Streamlit.

## Conclusion

This project demonstrates a complete end-to-end machine learning workflow, progressing from data cleaning and exploratory data analysis to feature selection, predictive modeling, and model evaluation.

The exploratory analysis provided insights into the demographic, personal, and workplace characteristics of employees, while feature selection techniques identified the variables most relevant for predicting treatment-seeking behavior. Building on these findings, three supervised classification models were developed and compared.

Among the evaluated models, the **Random Forest Classifier** achieved the best overall performance, with an **accuracy of 81.67%** and an **ROC-AUC score of 91.49%**, indicating excellent predictive capability.

Overall, this project demonstrates how machine learning can be applied to workplace mental health data to identify patterns associated with treatment-seeking behavior. While the models do not establish causal relationships, they provide valuable predictive insights that can help organizations better understand the factors associated with employees' decisions to seek professional mental health support and inform future workplace wellness initiatives.
