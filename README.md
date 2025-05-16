# Preventive Maintenance-ReneWind 

![image](https://github.com/user-attachments/assets/e7e744e6-a3cd-494a-acd3-47122e917501)


## Table of Contents

- [Project Background](#project-background)
- [Executive Summary](#executive-summary)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Preprocessing](#data-preprocessing)
- [Models Building](#models-building)
  - [Model building (original data)](#model-building-original-data)
  - [Model building oversampled data (SMOTE)](#model-building-oversampled-data-smote)
  - [Model building undersampled data](#model-building-undersampled-data)
- [Evaluation and Optimization ](#evaluation-and-optimization)
  - [Hyperparameter Tuning comparison](#hyperparameter-tuning-comparison)
  - [Final Model Evaluation on Test Set ](#final-model-evaluation-on-test-set)
  - [Future importances](#future-importances)
  - [Pipeline Evaluation](#pipeline-evaluation)
- [Insights](#insights)
- [Business Recommendations](#business-recommendations)
- [Assumptions & Limitations](#assumptions--limitations)
- [Setup Instructions](#setup-instructions)

## Project Background 

<div align="justify">
ReneWind is a company working on improving the machinery/processes involved in the production of wind energy using machine learning. As part of its commitment to sustainability and operational efficiency, the company aimed to reduce the high costs associated with unexpected generator failures, which often lead to unplanned downtime, emergency repairs, and full equipment replacements. I as part of the data sciences team partnered with the Preventive Maintenance and Engineering teams to build various classification models that predicts failures based on sensor data. The model prioritized minimizing false negatives to prevent costly breakdowns, while also reducing false positives to avoid unnecessary inspections. This solution enabled early failure detection and more efficient, data driven maintenance planning.
</div>

---

## Executive Summary

<div align="justify">
ReneWind’s analysis of 25,000 sensor based records from wind turbines led to the development of a predictive maintenance model aimed at preventing generator failures. Using XGBoost with oversampling, the final model achieved an F1-score of 83.4%, with 84.8% recall and 82% precision on test data successfully balancing failure detection and false alarm reduction. Key sensor variables, including V36, V26, and V16, were identified as the most influential predictors of failure. By implementing the model in collaboration with the maintenance team, ReneWind can detect up to 85 out of every 100 real failures before they occur, reduce maintenance costs, and extend the lifespan of critical turbine components, ultimately improving operational reliability and supporting long term sustainability goals.
</div>

---

## Exploratory Data Analysis (EDA)

<div align="justify">
The dataset contains 25,000 sensor based operational records from wind turbines, used to build a predictive model for generator failure. It includes 39 anonymized predictor variables and one target variable indicating failure events (1) or normal operation (0).
The data was split into 20,000 training and 5,000 testing rows. Due to confidentiality, variables were ciphered, though it is inferred that they represent readings from sensors such as temperature, vibration, and wind speed across key components (e.g., gearbox, blades, tower).
The data types are mostly floats, except for the target variable, which is an integer. Some missing values were present in both the training and test sets and will be addressed in the preprocessing stage discussed later. No duplicate entries were found.
</div>

EDA revealed frequent outliers across sensor readings likely reflecting early signs of mechanical anomalies rather than noise. Most variables show skewed but reasonably symmetric distributions, making the dataset suitable for predictive modeling.
</div>

<div align="center" style="display: flex; justify-content: center; gap: 20px;">
  <img src="https://github.com/user-attachments/assets/589f940a-786d-47b5-9c87-0aeb4d4b80c1" width="300"/>
  <img src="https://github.com/user-attachments/assets/80fbdd1f-f76a-459c-815d-36f34d89d451" width="700"/>
</div>


---

## Data Preprocessing 

**Missing values**

To handle missing values in both of the datasets, a median imputation strategy was applied using Scikit-learn’s **SimpleImputer**. The imputer was fitted on the training set to prevent data leakage, and the same transformation was then applied to the validation and test sets. **Median imputation** was chosen due to its robustness to outliers and its ability to maintain central tendency.

---

## Models Building 

To identify the most suitable algorithm for predicting generator failures, I trained and evaluated seven classification models Logistic **Regression, Bagging, Random Forest, Gradient Boosting, AdaBoost, XGBoost, and Decision Tree** across three datasets: the original imbalanced dataset, an oversampled version, and an undersampled version.

---

## Model building (original data) 

All seven classification models performed well in terms of recall, with consistent results across cross validation and validation sets indicating strong generalization capability.
Among them, **XGBoost** delivered the best performance, achieving the highest recall **83.7%** and showing the most stable results across folds. As shown in the boxplot, the highest recall was achieved by XGBoost, followed by Decision Tree and Random Forest

<p align="center">
  <img src="https://github.com/user-attachments/assets/453d9f32-56c6-4150-a951-6332f5576509" width="400"/>
  <img src="https://github.com/user-attachments/assets/54f4ea36-1357-4bee-b669-074a7ac52e39" width="500"/>
</p>

---

## Model building oversampled data (SMOTE)

To address the class imbalance in the training data **(833 failures vs. 14,167 non failures)**, I applied SMOTE to generate synthetic examples of the minority class. After oversampling, the training set was perfectly balanced, with 14,167 instances for each class, totaling 28,334

After applying oversampling, all models showed notable improvement especially those that previously struggled with class imbalance, such as **Logistic Regression and AdaBoost**. In this scenario, **Gradient Boosting** have the highest recall, closely followed by **XGBoost**. Gradient Boosting achieved a recall of **89.5%** on the validation set. Even the lowest performing model reached a recall of 81%, demonstrating the overall effectiveness of the oversampling strategy. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/731524f0-50f8-4d34-b7e2-22cc6f0032e6" width="400"/>
  <img src="https://github.com/user-attachments/assets/e6f4e198-d5d1-4e5f-80d5-19a9a0d6532f" width="400"/>
</p>

---

## Model building undersampled data

To address class imbalance from the opposite direction, I applied Random Undersampling, which reduced the number of majority class samples (non failures) to match the minority class (failures). This resulted in a balanced training set with 1,666 total observations (833 per class).
 
Among the three approaches tested (original, oversampled, and undersampled), the undersampled model delivered the best recall, reaching up to 89.8% on the validation set with the Gradient Boosting model. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/a12cc0d9-27bf-446e-904f-e71480df30d2" width="400"/>
  <img src="https://github.com/user-attachments/assets/efe62ff2-94a6-461d-bec6-a522d8a7071a" width="400"/>
</p>

---

## Evaluation and Optimization

 After evaluating the baseline performance of all models, I selected the top performing ones for further improvement through hyperparameter tuning. These models demonstrated strong recall scores and consistent validation performance.

## Hyperparameter Tuning comparison

<div align="justify">
In the final model, there was overfitting in the training performance. However, when comparing validation results, the best model achieved a **recall of 89%**, obtained with **XGBoost tuned on the oversampled dataset**. This model also reached the highest precision among the four tested models, with **82%**. The focus on recall and precision aligns with the project’s objective: to accurately detect true failures while minimizing false positives.
</div>


<div align="center">
  <img src="https://github.com/user-attachments/assets/f232afd5-04e1-43ac-b814-3a5e1e1e10ea" width="900"/>
</div>

 ---

## Final Model Evaluation on Test Set 

After training, tuning, and validating the XGBoost model using oversampled data, I performed a final evaluation on the unseen test set to confirm its generalization performance.

- The final model maintained high recall, effectively identifying most actual failures.

- Precision remained consistently strong, meaning few false alarms.

- With an F1-score of 83.7%, the model confirmed its ability to balance detection sensitivity and reliability on completely unseen data.

 <div align="center">
  <img src="https://github.com/user-attachments/assets/3c06118d-bd1a-4f8b-94e4-18152db2469a" width="400"/>
</div>

---

## Future importances

The final XGBoost model highlighted a small subset of sensor features as the most influential for predicting generator failures. The top five predictors V36, V26, V14, V16, and V18 stood out for their high relative importance.While feature names are anonymized due to confidentiality,they likely represent critical sensor readings related to turbine components or environmental factors such as temperature, pressure, or vibration.In other words, these features show strong patterns or behaviors that help the model detect failures in advance, making them key indicators for predictive maintenance.


 <div align="center">
  <img src="https://github.com/user-attachments/assets/670b3141-74c4-4537-a0b1-69d3ee27f744" width="400"/>
</div>

---

 ## Pipeline Evaluation

 To ensure model reproducibility and scalability, the final XGBoost classifier was wrapped in a pipeline that includes missing value imputation (median strategy) and SMOTE based oversampling. The pipeline was retrained and evaluated on the unseen test set.

 These results confirm the final model’s robustness in real world deployment scenarios, with strong generalization, reliable fault detection, and a controlled false positive rate.
 
  <div align="center">
  <img src="https://github.com/user-attachments/assets/acea22ab-8a3c-4653-b6e9-af04475dfb6e" width="400"/>
</div>

 ---

 ## Insights 

**The XGBoost model showed strong performance, achieving:**

- The final model achieved a **recall of 84.8%**, compared to an average of **64.2 %** across initial baseline models without sampling or tuning. This represents an improvement of approximately **1.3×** in failure detection, significantly increasing the model’s ability to identify real breakdowns before they occur.

- Alsow achieved a **precision of 82.1%**. This represents an improvement in reducing **false alarms**, enabling more efficient maintenance decisions and minimizing unnecessary interventions.

- An overall **F1-score of 83.4%**, reflecting a solid balance between detecting **real issue** and minimizing **false alarms** making the model effective and practical for the company objective.

**Five variables account for over 60% of model decision power:**

- The **future variables** are those with the highest relative importance in the model, suggesting they are the most **influential in predicting failures** (V36, V26, V16, V14, and V18). As such, they should be prioritized for **sensor calibration**, **data quality assurance**, and **future diagnostics**.

**The model generalized well to unseen data:**

- The model performed just as well on new, **unseen data** as it did during validation. This means it didn't just memorize the training data it actually **learned useful patterns** that apply to real world cases. Because of this consistency, the model is reliable enough to be tested in a pilot phase within operations.

---

## Business Recommendations

**Embed model output into the maintenance scheduling system:**

Use the model’s predictions to rank turbines by failure risk. Prioritize those in the top decile for manual inspection or remote diagnostics. This allows field engineers to act before failures escalate to major breakdowns.

**Integrate feedback from maintenance teams into the modeling process:**

- Actively collecting and reviewing technician feedback on the model’s predictions ( “false alarms” or missed failures) can help refine the system over time. Establishing this feedback loop allows the Data Science team to better align predictions with real world observations, adjust thresholds if needed, and continuously improve model relevance and adoption in the field.

- This also helps strengthen the collaboration between the Engineering, Maintenance, and Data Science teams by making it easier to understand the specific needs of each area. As a result, teams can stay aligned and work more efficiently toward shared operational goals.

**Refine Sensor Strategy:**

- Consider re evaluating sensor placement and maintenance on the most important features (V36, V26, etc.). Investigate whether sensors related to these features require more frequent calibration or if additional sensor types could enhance coverage.

**Expand the Model to Multi Failure Classification:**

Currently, the model performs binary classification to predict whether a failure will occur. While this provides valuable support for proactive maintenance planning, future iterations could significantly enhance operational impact by shifting to multi class classification predicting not only if a failure will happen, but also what type of failure ( gearbox, brake, generator).

This upgrade would enable:

- **Targeted maintenance actions**, tailored to the specific component at risk

- **Optimized spare parts inventory**, reducing overstock and emergency procurement

- **Minimized downtime**, by ensuring technicians arrive prepared with the right tools and parts

- **Improved operational planning**, especially for remote or offshore wind farms

---

## Assumptions & Limitations

**Missing values**

- Missing values in the **V1 and V2 variables** were addressed using **standard imputation techniques** during preprocessing. However, an **alternative approach** worth considering would be to **retain these values as missing**. This suggestion stems from the hypothesis that the **simultaneous absence** of V1 and V2 could carry **predictive value** potentially signaling a **system failure or malfunction**.

- Although this approach was **not implemented** in the current version of the model, it represents a **valuable direction for future iterations**. In such cases, leveraging algorithms like **XGBoost**, which **natively handle missing values**, could enable the model to **learn from the missingness itself as an informative signal**, rather than treating it as noise.

**EDA analysis**

The EDA analysis revealed that some histograms are significantly **skewed**. This raises the hypothesis that **highly skewed variables may be associated with failure rates or system breakdowns**, and therefore could be relevant predictors. These variables were flagged for closer attention in the modeling phase.

However, due to **confidentiality limitations** and the lack of context regarding the **specific types of sensors or features being analyzed**, it is difficult to validate this assumption. Without knowing what each variable represents, interpreting the skewness becomes speculative. 

---

## Setup Instructions

nstallation

To install the required libraries, run:

```bash
pip install -r requirements.txt




