#  ReneWind - Preventive Maintenance in Renewable Energy

## Table of Contents

- [Project Background](#project-background)
- [Executive Summary](#executive-summary)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Insights Deep Dive](#insights-deep-dive)
  - [Lead Time Impact](#lead-time-impact)
  - [Special Requests vs Cancellation](#special-requests-vs-cancellation)
  - [Online Booking Behavior](#online-booking-behavior)
  - [Price Sensitivity & Cancellation](#price-sensitivity--cancellation)
  - [Seasonality & Booking Reliability](#seasonality--booking-reliability)
  - [Loyalty vs First-Time Guests](#loyalty-vs-first-time-guests)
- [Modeling Approach](#modeling-approach)
  - [Logistic Regression](#logistic-regression)
  - [Decision Tree](#decision-tree)
  - [Threshold Optimization](#threshold-optimization)
- [Model Performance Summary](#model-performance-summary)
- [Business Recommendations](#business-recommendations)
- [Assumptions & Limitations](#assumptions--limitations)

## Project Background 

<div align="justify">
ReneWind is a renewable energy company specializing in the maintenance and operation of wind turbine infrastructure. As part of its commitment to sustainability and operational efficiency, the company aimed to reduce the high costs associated with unexpected generator failures, which often lead to unplanned downtime, emergency repairs, and full equipment replacements. I partnered with the Preventive Maintenance and Engineering teams to build a machine learning model that predicts failures based on sensor data. The model prioritized minimizing false negatives to prevent costly breakdowns, while also reducing false positives to avoid unnecessary inspections. This solution enabled early failure detection and more efficient, data driven maintenance planning.
</div>

---

## Executive Summary

<div align="justify">
ReneWind’s analysis of 25,000 sensor-based records from wind turbines led to the development of a predictive maintenance model aimed at preventing generator failures. Using XGBoost with oversampling, the final model achieved an F1-score of 86.0%, with 84.8% recall and 87.2% precision on test data—successfully balancing failure detection and false alarm reduction. Key sensor variables, including V36, V16, and V18, were identified as the most influential predictors of failure. By implementing the model in collaboration with the maintenance team, ReneWind can minimize unexpected breakdowns, reduce maintenance costs, and extend the lifespan of critical turbine components, ultimately improving operational reliability and supporting long-term sustainability goals.
</div>

---

## Exploratory Data Analysis (EDA)

<div align="justify">
The dataset contains sensor-based operational records from wind turbines and was used to build a predictive model for generator failure. Each row represents a single reading from various turbine components. The dataset includes 25,000 rows and 40 anonymized predictor variables, split into 20,000 for training and 5,000 for testing. Due to confidentiality, the data was provided in a ciphered format. Some missing values were found in both the training and test sets, but no duplicate entries were present.

Although the exact variables are anonymized, it can be inferred that the dataset includes readings from sensors such as temperature sensors, accelerometers, anemometers, and vibration monitors, capturing both environmental conditions (e.g., wind speed, humidity) and mechanical performance across components like the gearbox, blades, tower, and brake systems.
</div>

The EDA reveals that most of the sensors show a similar pattern in their boxplots: the presence of outliers on both tails of the distribution. These extreme sensor readings are common in real-world operational data and may contain valuable signals related to mechanical anomalies or early signs of failure, making them potentially informative rather than noise.

Additionally, the histograms show that most variables exhibit skewed distributions, yet many maintain a bell-shaped tendency. Despite the skewness, the data retains a reasonable degree of symmetry and central tendency, which makes it suitable for analysis.

<div align="center" style="display: flex; justify-content: center; gap: 20px;">
  <img src="https://github.com/user-attachments/assets/589f940a-786d-47b5-9c87-0aeb4d4b80c1" width="300"/>
  <img src="https://github.com/user-attachments/assets/80fbdd1f-f76a-459c-815d-36f34d89d451" width="700"/>
</div>


---

## Insights Deep Dive

Model building ( Original data) 

 ### Dataset Overview 

The dataset contains historical sensor readings from wind turbines. Each row represents a single turbine reading at a specific point in time, with the aim of identifying whether the unit is at risk of failure.

The dataset includes 25,000 rows in total, split into 20,000 observations for training and 5,000 for testing. It contains 40 anonymized predictor variables, which represent various environmental and mechanical parameters (such as temperature, vibration, torque, etc.), although the exact nature of the features is undisclosed due to data confidentiality.

The target variable is binary:

1 indicates a failure occurred.

0 indicates normal operation.

 
