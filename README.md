#  ReneWind - Preventive Maintenance in Renewable Energy

![image](https://github.com/user-attachments/assets/e7e744e6-a3cd-494a-acd3-47122e917501)


## Table of Contents

- [Project Background](#project-background)
- [Executive Summary](#executive-summary)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Building ](#model-building)
  - [Model building (original data)](#model-building-original-data)
  - [Model building oversampled data (SMOTE)](#model-building-oversampled-data-smote)
  - [Model building undersampled data](#model-building-undersampled-data)
- [Evaluation & Optimization ](#evaluation-&-optimization)
  - [Hyperparameter Tuning comparison](#hyperparameter-tuning-comparison)
  - [Final Model Evaluation on Test Set ](#final-model-evaluation-on-test-set)
  - [Future importances](#future-importances)
  - [Pipeline Evaluation](#pipeline-evaluation)
- [Insights](#insights)
- [Business Recommendations](#business-recommendations)
- [Assumptions & Limitations](#assumptions--limitations)

Evaluation & Optimization
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
The dataset contains 25,000 sensor-based operational records from wind turbines, used to build a predictive model for generator failure. It includes 39 anonymized predictors and 1 target variable indicating failure events (1) or normal operation (0).

The data was split into 20,000 training and 5,000 testing rows. Due to confidentiality, variables were ciphered, though it is inferred that they represent readings from sensors such as temperature, vibration, and wind speed across key components (e.g., gearbox, blades, tower). Some missing values were present in both the training and test sets, but no duplicate entries were found.
</div>

EDA revealed frequent outliers across sensor readings likely reflecting early signs of mechanical anomalies rather than noise. Most variables show skewed but reasonably symmetric distributions, making the dataset suitable for predictive modeling.
</div>

<div align="center" style="display: flex; justify-content: center; gap: 20px;">
  <img src="https://github.com/user-attachments/assets/589f940a-786d-47b5-9c87-0aeb4d4b80c1" width="300"/>
  <img src="https://github.com/user-attachments/assets/80fbdd1f-f76a-459c-815d-36f34d89d451" width="700"/>
</div>


---

##  Data Preprocessing 

**Missing values**

To handle missing values in both of the datasets, a median imputation strategy was applied using Scikit-learn’s **SimpleImputer**. The imputer was fitted on the training set to prevent data leakage, and the same transformation was then applied to the validation and test sets. **Median imputation** was chosen due to its robustness to outliers and its ability to maintain central tendency.

---

## Model Building 

## Model building (original data) 

To identify the most suitable algorithm for predicting generator failures, I trained and evaluated six classification models using the original (imbalanced) dataset: **Logistic Regression, Decision Tree, Random Forest, Bagging, AdaBoost, and XGBoost**. The main metric used for evaluation was **recall**, as the cost of missing a real failure (false negative) is significantly higher than flagging a non failure (false positive).

**Interpretation**:

XGBoost outperformed all other models, achieving the highest recall (83.8%) and the most consistent performance across cross validation. Its ability to handle noisy and imbalanced sensor data made it the most effective and reliable choice for detecting turbine failures.

<p align="center">
  <img src="https://github.com/user-attachments/assets/01ee90bc-abed-4a78-9961-176f43e1c44c" width="400"/>
  <img src="https://github.com/user-attachments/assets/0750b660-a17d-4894-8b63-848290e113d5" width="500"/>
</p>

---

## Model building oversampled data (SMOTE)

To address the class imbalance in the training data (833 failures vs. 14,167 non failures), I applied SMOTE to generate synthetic examples of the minority class. After oversampling, the training set was perfectly balanced, with 14,167 instances for each class, totaling 28,334

**Interpretation**:

XGBoost remained the best performing model after applying SMOTE. After oversampling, all models showed significant improvement—especially those that had previously struggled with imbalanced data, such as Logistic Regression. XGBoost continued to lead in performance, achieving the highest recall in both training (89.2%) and validation (99.0%), highlighting its robustness and strong generalization capability even after data augmentation.

<p align="center">
  <img src="https://github.com/user-attachments/assets/fa7832b0-c6fe-43b3-a0ef-cc6d3070b95a" width="400"/>
  <img src="https://github.com/user-attachments/assets/d27ae540-c3ce-4029-a7c4-c43c36661d86" width="400"/>
</p>

---

## Model building undersampled data

To address class imbalance from the opposite direction, I applied Random Undersampling, which reduced the number of majority class samples (non-failures) to match the minority class (failures). This resulted in a balanced training set with 1,666 total observations (833 per class).
The same six models were trained and evaluated using 5-fold cross-validation and tested on the same validation set using recall as the main performance metric.

**Interpretation**:

Random Forest was the best-performing model under undersampling, achieving the highest recall on the validation set (89.9%) and strong consistency during cross-validation. It handled the reduced dataset well without losing its predictive power, confirming its reliability for failure detection even with limited data.

<p align="center">
  <img src="https://github.com/user-attachments/assets/1adbb2a8-5888-46e5-913c-a9e43cbe9c67" width="400"/>
  <img src="https://github.com/user-attachments/assets/c7d3dc65-2944-4d5c-aed3-3c6801002a5b" width="400"/>
</p>

---

### Evaluation & Optimization

## Hyperparameter Tuning comparison

After benchmarking the baseline models, I performed hyperparameter tuning using RandomizedSearchCV on selected algorithms that showed strong performance: AdaBoost, Random Forest, Gradient Boosting, and XGBoost. The goal was to further enhance recall and F1-score, especially under oversampled and undersampled settings.

<div align="center">
  <img src="https://github.com/user-attachments/assets/e5ae5213-8e66-4e95-b712-68d26d7f87fa" width="800"/>
</div>

<br>

<div align="center">
  <img src="https://github.com/user-attachments/assets/6d23ea56-3461-4c05-82f6-b70c41f34b74" width="800"/>
</div>

**Interpretation**

 XGBoost tuned with oversampled data (SMOTE) offered the best overall balance between recall and precision, achieving an F1-score of 0.883 on the validation set. This means it was not only able to detect most failure events (high recall), but also minimized false alarms (high precision), which is crucial in maintenance operations.

 ---

## Final Model Evaluation on Test Set 

After training, tuning, and validating the XGBoost model using oversampled data, I performed a final evaluation on the unseen test set to confirm its generalization performance.

 <div align="center">
  <img src="https://github.com/user-attachments/assets/ad932029-ebd1-4bda-942f-dd04867e7a7f" width="400"/>
</div>

- The final model maintained high recall, effectively identifying most actual failures.

- Precision remained consistently strong, meaning few false alarms.

- With an F1-score of 85.8%, the model confirmed its ability to balance detection sensitivity and reliability on completely unseen data.

---

## Future importances

The final XGBoost model highlighted a small subset of sensor features as the most influential for predicting generator failures. The top five predictors V36, V16, V18, V26, and V14 stood out for their high relative importance.While feature names are anonymized due to confidentiality,they likely represent critical sensor readings related to turbine components or environmental factors such as temperature, pressure, or vibration.In other words, these features show strong patterns or behaviors that help the model detect failures in advance, making them key indicators for predictive maintenance.


 <div align="center">
  <img src="https://github.com/user-attachments/assets/dd97c911-cb3f-4688-b323-94b625ec5b68" width="400"/>
</div>

---

 ## Pipeline Evaluation

 To ensure model reproducibility and scalability, the final XGBoost classifier was wrapped in a pipeline that includes missing value imputation (median strategy) and SMOTE-based oversampling. The pipeline was retrained and evaluated on the unseen test set.

 These results confirm the final model’s robustness in real-world deployment scenarios, with strong generalization, reliable fault detection, and a controlled false positive rate.
 
  <div align="center">
  <img src="https://github.com/user-attachments/assets/97877d51-c14a-463d-ba25-05b3b4d1499d" width="400"/>
</div>

 ---

 ## Insights 

**The XGBoost model showed strong performance, achieving:**

- An **accuracy of 98.4%**, meaning it made correct predictions in nearly all cases.

- A **recall of 84.8%**, correctly detecting approximately **85 out of every 100 actual failures**, helping prevent unexpected breakdowns.

- A **precision of 87.2%**, indicating that when the model predicts a failure, it is correct **87 times out of 100**, reducing unnecessary maintenance actions.

- An overall **F1-score of 86.0%**, reflecting a solid balance between detecting **real issue** and minimizing **false alarms** making the model effective and practical for the company objective. 
**Five variables account for over 60% of model decision power:**

- The **future variables** are those with the highest relative importance in the model, suggesting they are the most **influential in predicting failures** (V36, V16, V18, V26, and V14).
  
- These features exhibit strong patterns that allow the model to detect failures in advance, making them **key indicators** for predictive maintenance. As such, they should be prioritized for **sensor calibration**, **data quality assurance**, and **future diagnostics**.

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

- Consider re evaluating sensor placement and maintenance on the most important features (V36, V16, etc.). Investigate whether sensors related to these features require more frequent calibration or if additional sensor types could enhance coverage.

**Expand the Model to Multi Failure Classification:**

Currently, the model performs binary classification to predict whether a failure will occur. While this provides valuable support for proactive maintenance planning, future iterations could significantly enhance operational impact by shifting to multi class classification predicting not only if a failure will happen, but also what type of failure ( gearbox, brake, generator).

This upgrade would enable:

- **Targeted maintenance actions**, tailored to the specific component at risk

- **Optimized spare parts inventory**, reducing overstock and emergency procurement

- **Minimized downtime**, by ensuring technicians arrive prepared with the right tools and parts

- **Improved operational planning**, especially for remote or offshore wind farms
