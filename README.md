# M.Tech (AIML) Machine Learning — Assignment 2 Documentation

### Student & Submission Profile
* **Course Frame:** M.Tech (Artificial Intelligence & Machine Learning)
* **Module Identifier:** Machine Learning (Assignment - 2)
* **Execution Space:** BITS Virtual Lab Environment

---

## a. Problem Statement
The operational focus of this framework is to establish highly reliable automated diagnostics for cellular tissue screening classifications. By processing multi-dimensional spatial diagnostic matrices derived from digitized images of fine needle aspirates (FNA) of breast masses, the deployed pipeline is designed to accurately evaluate whether diagnostic samples display signs of malignant growth variants or represent benign pathologies.

## b. Dataset Description
* **Database Catalog:** Breast Cancer Wisconsin (Diagnostic) Dataset (UCI Repository Source)
* **Sample Count (Instances):** 569 instances
* **Dimensional Vector Count (Features):** 30 distinct continuous real-valued spatial features tracking nuclear characteristics (e.g., perimeter radius scale variances, local uniformity attributes, texture roughness coefficients, concavity metrics).
* **Target Class Profile:** Binary output (0 representing Benign pathology, 1 representing Malignant pathology tracking profiles).

## c. GitHub Repository Link
* **Codebase Home:** `https://github.com/ketan0619/2026AJ05429-ML-Assignment-2.git`
* **Live Streamlit App:** `https://<YOUR_APP_SUBDOMAIN>.streamlit.app`

---

## d. Model Evaluation Metric Profiles

### Model Suite Performance Summary

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Decision Tree** | 0.9650 | 0.9716 | 0.9885 | 0.9556 | 0.9718 | 0.9268 |
| **KNN** | 0.9860 | 0.9998 | 0.9889 | 0.9889 | 0.9889 | 0.9700 |
| **Naive Bayes** | 0.9650 | 0.9950 | 0.9670 | 0.9778 | 0.9724 | 0.9249 |
| **Support Vector Machine**| 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### Observations on Model Performance

* **Logistic Regression:** Achieved perfect classification performance ($1.0000$ across all indicators) on this specific testing holdout partition. This shows that the standard feature scaling successfully mapped the spatial dimensions, enabling the model to draw a clean, linearly separable decision boundary without requiring complex boundary curves.
* **Decision Tree:** Registered solid performance metrics, including an accuracy of 96.50% and a high precision of 98.85%. Its slightly lower recall rate (95.56%) indicates that its recursive, axis-aligned splitting paths created a few rigid hyper-rectangular boundaries, missing a tiny fraction of edge-case malignant instances.
* **KNN Classifier:** Performed exceptionally well, delivering a 98.60% overall accuracy along with an outstanding AUC score of 0.9998. The distance-based neighborhood algorithm proved highly precise at handling localized geometry, clustering continuous diagnostic vectors with minimal confusion.
* **Naive Bayes:** Produced steady, robust classification tracking with an accuracy of 96.50% and a high AUC score of 0.9950. Even though it operates under a strict feature-independence assumption, it efficiently estimated class conditional probabilities across the high-dimensional input vectors.
* **Support Vector Machine (SVM):** Tied for top overall honors, achieving flawless scores ($1.0000$ across Accuracy, Recall, and MCC). The non-linear kernel projection successfully mapped the continuous spatial attributes into a distinct feature layout, allowing the structural margin hyperplane to separate the diagnostic categories perfectly.

### Overall Winner for the Dataset
The top execution spots are shared by **Logistic Regression** and the **Support Vector Machine (SVM)**, which both recorded perfect baseline metrics across the test data tracking framework. 

For the final architecture selection, **Logistic Regression** is designated as the overall structural winner. While both models achieved flawless predictive precision, Logistic Regression is computationally lighter, trains faster, requires significantly less memory overhead, and provides highly transparent, log-odds feature coefficients. This makes it the most practical, reliable, and production-ready option for real-world deployment inside a clinical dashboard environment.


