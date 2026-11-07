import streamlit as pd_app
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

# App Title & Context Setup
pd_app.set_page_config(page_title="M.Tech AIML: ML Assignment-2 Deployment Hub", layout="wide")
pd_app.title("📊🩺  Diagnostic Classification & Performance Diagnostics Dashboard")
pd_app.markdown("### Machine Learning - Assignment 2 (2026aj05429)")

# Sidebar for controls
pd_app.sidebar.header("🕹️ Control Panels")

# 1. Upload Option
uploaded_file = pd_app.sidebar.file_uploader("Upload Evaluation CSV Data", type=["csv"])

# 2. Model Selection Dropdown
chosen_algorithm = pd_app.sidebar.selectbox(
    "Select Classifier Model",
    ["Logistic_Regression", "Decision_Tree", "KNN", "Naive_Bayes", "Support_Vector_Machine"]
)

# Helper function to load model elements securely
def get_model_files(algo_name):
    scaler_path = "model/feature_scaler.pkl"
    model_path = f"model/{algo_name}_classifier.pkl"
    
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        return None, None
        
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return scaler, model

if uploaded_file is not None:
    try:
        # Load the user-submitted dataset
        evaluation_data = pd.read_csv(uploaded_file)
        
        # Verification check to see if target column exists
        target_col = 'diagnosis_target'
        if target_col not in evaluation_data.columns:
            pd_app.error(f"Missing ground truth tracking field: '{target_col}' in uploaded file.")
        else:
            # Display uploaded framework metrics slice
            pd_app.subheader("🔍 Sample of Uploaded Data Matrix")
            pd_app.dataframe(evaluation_data.head(6))
            
            # Split features and target labels
            X_eval = evaluation_data.drop(columns=[target_col])
            y_eval = evaluation_data[target_col]
            
            # Fetch backend model architectures
            scaler, classifier = get_model_files(chosen_algorithm)
            
            if classifier is not None:
                # Transform variables via pre-fitted scaler structure
                X_eval_scaled = scaler.transform(X_eval)
                
                # Perform pure feed-forward evaluations without retraining models
                predicted_labels = classifier.predict(X_eval_scaled)
                predicted_probs = classifier.predict_proba(X_eval_scaled)[:, 1]
                
                # Metric Assessments
                acc = accuracy_score(y_eval, predicted_labels)
                auc = roc_auc_score(y_eval, predicted_probs)
                prec = precision_score(y_eval, predicted_labels)
                rec = recall_score(y_eval, predicted_labels)
                f1 = f1_score(y_eval, predicted_labels)
                mcc = matthews_corrcoef(y_eval, predicted_labels)
                
                # Presentation Layout Structures
                pd_app.markdown(f"## ⚙️ Performance Framework: **{chosen_algorithm.replace('_', ' ')}**")
                
                metric_slots = pd_app.columns(6)
                metric_slots[0].metric("Accuracy Score", f"{acc:.4f}")
                metric_slots[1].metric("AUC Score", f"{auc:.4f}")
                metric_slots[2].metric("Precision Score", f"{prec:.4f}")
                metric_slots[3].metric("Recall Score", f"{rec:.4f}")
                metric_slots[4].metric("F1 Value", f"{f1:.4f}")
                metric_slots[5].metric("MCC Score", f"{mcc:.4f}")
                
                # Analysis splits: Reports vs Matrix Plots
                with col_left:
                      pd_app.subheader("📋 Class-Wise Performance Report")
    
                      report_dict = classification_report(
                            y_eval, predicted_labels, 
                            target_names=["Benign", "Malignant"], 
                            output_dict=True
                      )
  
                      report_df = pd.DataFrame(report_dict).transpose().round(2)
  
                      report_df.columns = ["Precision", "Recall", "F1-Score", "Support"]

                      report_df["Support"] = report_df["Support"].astype(int)
  
                      pd_app.dataframe(report_df, use_container_width=True)

                    
                with col_right:
                    pd_app.subheader("🧱 Confusion Matrix Blueprint Mapping")
                    conf_mat = confusion_matrix(y_eval, predicted_labels)
                    
                    fig, ax = plt.subplots(figsize=(4, 3.5))
                    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Purples", 
                                xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"], ax=ax, cbar=False)
                    plt.ylabel('Actual Label Class')
                    plt.xlabel('Model Predicted Class')
                    pd_app.pyplot(fig)
            else:
                pd_app.error("Model file parsing exception. Please check your tracking storage assets in the 'model/' subdirectory.")
                
    except Exception as general_err:
        pd_app.error(f"Processing error encounter: {general_err}")
else:
    pd_app.info("👋 Awaiting data ingestion sequence. Please drag or select the 'test_data.csv' file generated inside the sidebar area.")

