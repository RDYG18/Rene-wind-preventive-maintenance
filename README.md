#  ReneWind - Preventive Maintenance in Renewable Energy

## Table of Contents

- [Project Background](#project-background)
- [Executive Summary](#executive-summary)
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

## Project background 

<div align="justify">
ReneWind is a company specialized in maintaining infrastructure for renewable energy generation, with a primary focus on wind turbines. The operations team had identified that generator failures led to significant costs due to unexpected replacements, unplanned downtime, and emergency inspections.

As part of the Data Science team at ReneWind, I was tasked with supporting the Preventive Maintenance department by developing a predictive maintenance model. Our team worked closely with operations and engineering to leverage sensor data collected from various turbine components in order to analyze performance patterns and predict potential failures.
</div>

## Company Objective 

<div align="justify">
ReneWind’s objective was to build a classification model capable of predicting generator failures in wind turbines before they occurred, using historical sensor data.

The goal was to identify meaningful patterns that differentiated between at risk generators and those operating under normal conditions. Given the operational impact, I prioritized minimizing false negatives, as failing to detect an actual malfunction could result in costly generator replacements. At the same time, reducing false positives was also important to avoid unnecessary inspections and inefficient resource allocation.

The final model was developed to support the Preventive Maintenance team by flagging high-risk turbines early, enabling timely interventions and more efficient maintenance planning.
</div>

---

## Data Structure

 ### Dataset Overview 

The dataset contains historical sensor readings from wind turbines. Each row represents a single turbine reading at a specific point in time, with the aim of identifying whether the unit is at risk of failure.

The dataset includes 25,000 rows in total, split into 20,000 observations for training and 5,000 for testing. It contains 40 anonymized predictor variables, which represent various environmental and mechanical parameters (such as temperature, vibration, torque, etc.), although the exact nature of the features is undisclosed due to data confidentiality.

The target variable is binary:

1 indicates a failure occurred.

0 indicates normal operation.

 
