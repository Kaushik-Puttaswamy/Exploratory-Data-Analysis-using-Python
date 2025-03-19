# 📘 Exploratory-Data-Analysis-using-Python

## 🔍 Introduction

This project focuses on analyzing the performance of electronic components using Python. The tasks include data storage and retrieval, exploratory data analysis (EDA), model selection, data collection, data cleansing, data visualization, data mapping, deviation calculation, and unit testing. Three datasets are provided:

• 📂 Train dataset (used to select ideal functions)

• 📂 Ideal dataset (contains 50 ideal functions)

• 📂 Test dataset (used for validation and mapping)

This analysis helps predict equipment failures by comparing real-world component performance against optimal readings.

## 🎯 Problem Definition

The main task is to develop a Python program that:

1. 🏆 Selects the four best-fitting functions from 50 available ideal functions using training data.

2. 🔗 Uses the test data to map x-y pairs to one of the four ideal functions.

3. 📊 Stores the mapping results along with deviation calculations.

## 🎯 Aim and Objectives

The project aims to create a reliable and efficient Python program that:

• 🏆 Selects four ideal functions based on least squares error.

• 🔗 Maps test data to these ideal functions while considering deviation constraints.

• 📈 Evaluates performance using R-squared values and other error metrics.

## ❓ Research Questions

• 📌 How can we obtain the four best-fit ideal functions using the least squares method?

• 📌 What are the best alternative evaluation metrics for selection?

• 📌 Do alternative metrics yield the same ideal function choices as the least squares method?

• 📌 What are the R-squared values for the selected functions with test data?

• 📌 How does deviation change after mapping test data to ideal functions?

## 🏗 Structure of the Study

This research is structured into three main sections:

• 📖 Introduction: Overview, problem definition, objectives, research questions.

• 🛠 Investigation Method: EDA, database storage, function selection, and mapping.

• 📌 Conclusion: Summary of results, future scope, and recommendations.

## 📊 Exploratory Data Analysis (EDA)

EDA techniques were applied to analyze dataset properties and relationships. Various visualizations were used, including box plots, scatter plots, and correlation matrices.

### 📂 Train Dataset Analysis

• 📊 Boxplot of Train Dataset

![Boxplot of Train Dataset.png](https://github.com/Kaushik-Puttaswamy/Exploratory-Data-Analysis-using-Python/blob/main/Images/Boxplot%20of%20Train%20Dataset.png)

• 📈 Scatter plot with Regression Line

![Scatter plot with Regression Line.png](https://github.com/Kaushik-Puttaswamy/Exploratory-Data-Analysis-using-Python/blob/dev/Images/Scatter%20plot%20with%20Regression%20Line.png)

### 📂 Ideal Dataset Analysis

• 📊 Boxplot of Ideal Dataset

![Boxplot of Ideal Dataset.png](https://github.com/Kaushik-Puttaswamy/Exploratory-Data-Analysis-using-Python/blob/dev/Images/Boxplot%20of%20Ideal%20Dataset.png)

### 📂 Test Dataset Analysis

• 📊 Boxplot After Removing Duplicates

![Boxplot After Removing Duplicates.png](https://github.com/Kaushik-Puttaswamy/Exploratory-Data-Analysis-using-Python/blob/dev/Images/Boxplot%20After%20Removing%20Duplicates.png)

### 🗄 Data Storage and Retrieval

SQLite was used for storing datasets, accessed via SQLAlchemy ORM in Python.

#### 📌 Database Schema

🏷 Training Data Table

🏷 Ideal Functions Table

🏷 Test Data Mapping Table

## 🔎 Finding Ideal Functions

📐 Least Squares Analysis

Bar charts were generated for each function:

• 📊 Least Squares Bar Chart (Y1 Train Data)

![Least Squares Bar Chart (Y1 Train Data).png](https://github.com/Kaushik-Puttaswamy/Exploratory-Data-Analysis-using-Python/blob/dev/Images/Least%20Squares%20Bar%20Chart%20(Y1%20Train%20Data).png)

• 📊 Least Squares Bar Chart (Y2 Train Data)

![Least Squares Bar Chart (Y2 Train Data).png](https://github.com/Kaushik-Puttaswamy/Exploratory-Data-Analysis-using-Python/blob/dev/Images/Least%20Squares%20Bar%20Chart%20(Y2%20Train%20Data).png)

• 📊 Least Squares Bar Chart (Y3 Train Data)

![Least Squares Bar Chart (Y3 Train Data).png](https://github.com/Kaushik-Puttaswamy/Exploratory-Data-Analysis-using-Python/blob/dev/Images/Least%20Squares%20Bar%20Chart%20(Y3%20Train%20Data).png)

• 📊 Least Squares Bar Chart (Y4 Train Data)

![Least Squares Bar Chart (Y4 Train Data).png](https://github.com/Kaushik-Puttaswamy/Exploratory-Data-Analysis-using-Python/blob/dev/Images/Least%20Squares%20Bar%20Chart%20(Y4%20Train%20Data).png)

## 📈 Regression Line Comparisons

• Scatter plot of Training vs. Ideal Functions

![Scatter plot of Training vs. Ideal Functions.png](https://github.com/Kaushik-Puttaswamy/Exploratory-Data-Analysis-using-Python/blob/dev/Images/Scatter%20plot%20of%20Training%20vs.%20Ideal%20Functions.png)

## 📉 Mean Squared Error Method

• 📊 R-Squared Values for Test Data Mapping

![R-Squared Values for Test Data Mapping.png](https://github.com/Kaushik-Puttaswamy/Exploratory-Data-Analysis-using-Python/blob/dev/Images/R-Squared%20Values%20for%20Test%20Data%20Mapping.png)

## 🔗 Mapping Test Dataset with Ideal Functions

• 📊 Absolute Maximum Deviation Bar Chart

![Absolute Maximum Deviation Bar Chart_1.png](https://github.com/Kaushik-Puttaswamy/Exploratory-Data-Analysis-using-Python/blob/dev/Images/Absolute%20Maximum%20Deviation%20Bar%20Chart_1.png)

• 📈 Scatter Plot of Mapped Test Data

![Scatter Plot of Mapped Test Data.png](https://github.com/Kaushik-Puttaswamy/Exploratory-Data-Analysis-using-Python/blob/dev/Images/Scatter%20Plot%20of%20Mapped%20Test%20Data.png)

## 🏁 Conclusion

The project successfully developed a Python program that:

• ✅ Selected the four best-fitting functions using least squares error.

• 🔗 Mapped test data points to these ideal functions.

• 📊 Evaluated the deviation between actual and ideal data.

## 🚀 Future Scope

• 🔎 Investigate alternative evaluation metrics.

• 📈 Improve accuracy using advanced machine learning techniques.

• 🤖 Automate parameter tuning for better function selection.

