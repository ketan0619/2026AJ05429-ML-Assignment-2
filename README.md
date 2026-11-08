# M.Tech (AIML) Machine Learning — Assignment 2 Documentation

### Student Profile
* **Name:** Ketan Hanumant Gargote
* **BITS-ID:** 2026AJ05429
* **Course:** M.Tech (Artificial Intelligence & Machine Learning)
* **Assignment:** Machine Learning (Assignment - 2)
* **Environment:** BITS Virtual Lab

---

## a. Problem Statement
The goal of this project is to build a machine learning pipeline that can classify breast mass samples as either malignant or benign. By training multiple classification models on clinical feature data, we aim to find the best algorithm to deploy as an interactive web dashboard for quick diagnostic support.

## b. Dataset Description
* **Database Used:** Breast Cancer Wisconsin (Diagnostic) Dataset from the UCI ML Repository.
* **Total Rows (Instances):** 569
* **Total Columns (Features):** 30 continuous features (such as radius, texture, perimeter, area, smoothness, and concavity of the cell nuclei).
* **Target Variable:** Binary classification column named diagnosis_target (where 0 means Benign and 1 means Malignant).

## c. GitHub Repository Link
* **Repo Link:** `https://github.com/ketan0619/2026AJ05429-ML-Assignment-2.git`
* **Live Streamlit App:** `https://2026aj05429-ml-assignment-2-upjmb34jz4diye4eyhds9a.streamlit.app/`

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

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression:** |This model gave a perfect score of 1.00 across all metrics on our test split. Because the data was scaled using StandardScaler, the model was able to easily find a clean linear separation between the classes. During execution in the BITS lab, a SAGA convergence warning was triggered, showing that the model hit its max iterations limit, but it still managed to find an optimal decision boundary. |
| **Decision Tree:** | Hit an accuracy of 96.50%. It has high precision (98.85%) but a slightly lower recall of 95.56%, meaning it missed a couple of malignant cases. This happens because a simple decision tree creates rigid splits and can struggle slightly on continuous data boundaries compared to distance-based models. |
| **KNN Classifier:** | Performed exceptionally well with 98.60% accuracy and an AUC of 0.9998. Using n_neighbors=7 along with distance weighting worked perfectly for this geometry, keeping false positives and false negatives very low (both precision and recall are at 98.89%). |
| **Naive Bayes:** | Got a solid 96.50% accuracy and 0.9950 AUC. Even though it assumes all 30 features are completely independent (which isn't strictly true for things like cell radius and perimeter), it still gave a highly reliable and fast classification. |
| **Support Vector Machine (SVM):** | Like Logistic Regression, the SVM model scored a perfect 1.00 on this test split. The RBF kernel mapped the scaled continuous features perfectly. The lab environment gave a future warning about the probability parameter being deprecated in newer scikit-learn versions, but the model still saved and serialized correctly. |
| **Overall Winner for my Dataset:** | The top models are Logistic Regression and SVM, since both achieved a flawless 1.00 score on the test set. For the final deployment choice, I am selecting Logistic Regression as the winner. Even though the predictions match SVM exactly, Logistic Regression is computationally lighter, uses less memory, and processes requests faster inside the Streamlit app. It also provides highly clear feature weights, making it much easier to interpret from a medical standpoint than a complex SVM hyperplane. |



