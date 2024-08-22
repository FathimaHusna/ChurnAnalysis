# Telecom Churn Analysis

![Churn Analysis Visualization](intro.jpg "image source")

## Overview

Churn analysis is crucial for telecom companies to understand and mitigate customer attrition, which directly impacts revenue and customer loyalty. This project focuses on analyzing customer data to predict churn and uncover the factors driving customer departure. The dataset includes various attributes related to customer demographics, service usage, and billing information, providing a comprehensive basis for analysis.

## Objectives

1. **Data Exploration:** Thoroughly explore the dataset to identify patterns and factors associated with customer churn.
2. **Feature Engineering:** Develop and optimize features to enhance the accuracy of churn prediction models.
3. **Modeling:** Build and evaluate machine learning models to predict customer churn with high accuracy.
4. **Insights:** Identify the most significant factors influencing customer churn and derive actionable insights.

## Dataset

The dataset includes the following features:

- **customerID:** A unique identifier for each customer.
- **gender:** Gender of the customer (e.g., Female, Male).
- **SeniorCitizen:** Whether the customer is a senior citizen (1 if yes, 0 if no).
- **Partner:** Whether the customer has a partner (Yes/No).
- **Dependents:** Whether the customer has dependents (Yes/No).
- **tenure:** Number of months the customer has been with the company.
- **PhoneService:** Whether the customer subscribes to phone service (Yes/No).
- **MultipleLines:** Whether the customer has multiple lines (No phone service, Single line, Multiple lines).
- **InternetService:** Type of internet service (e.g., DSL, Fiber optic, No).
- **OnlineSecurity:** Whether the customer has online security (Yes/No).
- **OnlineBackup:** Whether the customer has online backup (Yes/No).
- **DeviceProtection:** Whether the customer has device protection (Yes/No).
- **TechSupport:** Whether the customer has tech support (Yes/No).
- **StreamingTV:** Whether the customer streams TV (Yes/No).
- **StreamingMovies:** Whether the customer streams movies (Yes/No).
- **Contract:** Type of customer contract (e.g., Month-to-month, One year, Two year).
- **PaperlessBilling:** Whether the customer has paperless billing (Yes/No).
- **PaymentMethod:** Payment method used by the customer (e.g., Electronic check, Mailed check, Bank transfer, Credit card).
- **MonthlyCharges:** Monthly charges billed to the customer.
- **TotalCharges:** Total charges incurred by the customer.
- **Churn:** Whether the customer has churned (Yes/No).

## Steps Involved

### 1. Data Preprocessing

- **Import Libraries:** Import necessary libraries for data manipulation and visualization.
- **Load Dataset:** Load the telecom churn dataset into a DataFrame.
- **Initial Data Inspection:** View the first few rows, check the shape of the dataset, and inspect column names.
- **Handle Missing Values:** Identify and handle any missing values in the dataset.
- **Convert Data Types:** Ensure that all columns are of the correct data type.
- **Encode Categorical Variables:** Convert categorical variables into numerical format.
- **Normalize/Standardize Features:** Scale numerical features to ensure consistent ranges.

### 2. Exploratory Data Analysis (EDA)

- **Feature Distribution Visualization:** Plot distributions of various features to understand their spread and detect any anomalies.
- **Churn Distribution:** Visualize the distribution of the target variable `Churn`.
- **Correlation Analysis:** Generate a correlation matrix to understand relationships between features.
- **Univariate Analysis:** Examine each feature individually with respect to `Churn`.
- **Bivariate Analysis:** Explore relationships between pairs of variables to gain insights.

### 3. Model Development

- **Data Splitting:** Split the dataset into training and testing sets.
- **Apply SMOTE:** Use the SMOTE technique to handle class imbalance in the training data.
- **Model Training:** Train multiple machine learning models, such as Decision Tree and Random Forest.
- **Model Evaluation:** Evaluate models using metrics like accuracy, precision, recall, F1-score, and confusion matrices.

