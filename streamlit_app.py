import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, average_precision_score, 
                           accuracy_score, f1_score, precision_score, recall_score, auc)

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# --- Load Data ---
@st.cache_data()
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        data_dict = {
            'Id': 'Unique identifier for each loan application',
            'Age': 'Age of the applicant',
            'Income': 'Annual income of the applicant',
            'Home': 'Home ownership status (RENT, OWN, MORTGAGE, OTHER)',
            'Emp_length': 'Employment length in months',
            'Intent': 'Loan intent (PERSONAL, EDUCATION, MEDICAL, VENTURE, etc.)',
            'Amount': 'Loan amount requested',
            'Rate': 'Interest rate on the loan',
            'Status': 'Loan status (binary)',
            'Percent_income': 'Loan amount as percentage of income',
            'Default': 'Target variable - whether loan defaulted (Y/N)',
            'Cred_length': 'Length of credit history in years'
        }
        st.subheader("Data Validation")
        st.write(f"Dataset shape: {df.shape}")
        missing_values = df.isnull().sum()
        st.write("Missing values:")
        st.dataframe(missing_values[missing_values > 0])
        st.write(f"Duplicate rows: {df.duplicated().sum()}")
        df['Rate'] = df.groupby('Intent')['Rate'].transform(lambda x: x.fillna(x.median()))
        df['Income_quartile'] = pd.qcut(df['Income'], q=4, labels=False)
        df['Emp_length'] = df.groupby('Income_quartile')['Emp_length'].transform(lambda x: x.fillna(x.median()))
        df.drop('Income_quartile', axis=1, inplace=True)
        df['Default'] = df['Default'].astype(str).str.strip().str.lower().map({
            'y': 1, 'yes': 1, '1': 1,
            'n': 0, 'no': 0, '0': 0
        })
        df['Default'].fillna(0, inplace=True)
        return df, data_dict
    return None, None

def create_features(df):
    df = df.copy()
    df['debt_to_income'] = df['Amount'] / df['Income']
    df['age_bucket'] = pd.cut(df['Age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle-aged', 'Senior'], include_lowest=True)
    df['loan_to_income'] = df['Amount'] / df['Income']
    df['rate_income_interaction'] = df['Rate'] * df['Percent_income']
    return df

def load_model(model_path):
    return joblib.load(model_path)

def predict(model, data):
    y_pred = model.predict(data)
    if isinstance(y_pred[0], str):
        y_pred = pd.Series(y_pred).str.strip().str.lower().map({'y': 1, 'yes': 1, 'n': 0, 'no': 0, '0': 0, '1': 1}).values
    y_pred_proba = model.predict_proba(data)[:, 1]
    return y_pred, y_pred_proba

def plot_roc_curve(y_true, y_pred_proba, model_name="Model"):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    return fig

def plot_precision_recall_curve(y_true, y_pred_proba, model_name="Model"):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, lw=2, label=f'{model_name} (AP = {avg_precision:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left")
    return fig

def show_metrics(y_true, y_pred, y_pred_proba, model_name="Model"):
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.subheader(f"{model_name} Metrics")
    st.dataframe(df_report.style.format("{:.2f}"))
    st.subheader("Confusion Matrix")
    conf_matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Non-Default', 'Default'],
                yticklabels=['Non-Default', 'Default'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)
    st.write(f"ROC AUC: {roc_auc_score(y_true, y_pred_proba):.3f}")
    st.write(f"PR AUC: {average_precision_score(y_true, y_pred_proba):.3f}")

def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title("Feature Importances")
        ax.barh(range(len(indices)), importances[indices], color='b', align='center')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.invert_yaxis()
        return fig
    return None

def plot_shap_values(model, X, feature_names):
    try:
        if isinstance(model, Pipeline):
            transformer_steps = model.steps[:-1]
            transformer = Pipeline(transformer_steps)
            X_transformed = transformer.transform(X)
            final_estimator = model.steps[-1][1]
        else:
            X_transformed = X
            final_estimator = model
        explainer = shap.Explainer(final_estimator)
        shap_values = explainer.shap_values(X_transformed)
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, plot_type="bar")
        return fig
    except Exception as e:
        st.warning(f"SHAP visualization not available for this model type. Error: {str(e)}")
        return None

def main():
    st.title("Loan Default Prediction Dashboard")
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the section", ["Data Overview", "Exploratory Analysis", "Model Prediction"])
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
    df, data_dict = load_data(uploaded_file)
    if df is not None:
        if app_mode == "Model Prediction":
            X_full = df.drop(columns=['Default'])
            X = create_features(X_full.drop(columns=['Id'], errors='ignore'))
            y = df['Default']
            cat_features = ['Home', 'Intent', 'age_bucket']
            num_features = ['Age', 'Income', 'Emp_length', 'Amount', 'Rate', 
                            'Percent_income', 'Cred_length', 'debt_to_income', 
                            'loan_to_income', 'rate_income_interaction']
            model_choice = st.selectbox("Select a Model", ["Logistic Regression", "Random Forest", "Gradient Boosting"])
            model_path = ""
            if model_choice == "Logistic Regression":
                model_path = "best_credit_risk_model.pkl"
            elif model_choice == "Random Forest":
                model_path = "random_forest_pipeline.pkl"
            elif model_choice == "Gradient Boosting":
                model_path = "gradient_boosting_pipeline.pkl"
            if model_path:
                try:
                    model = load_model(model_path)
                    st.subheader("Model Predictions")
                    y_pred, y_pred_proba = predict(model, X_full)
                    show_metrics(y, y_pred, y_pred_proba, model_choice)
                    st.subheader("Model Performance")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(plot_roc_curve(y, y_pred_proba, model_choice))
                    with col2:
                        st.pyplot(plot_precision_recall_curve(y, y_pred_proba, model_choice))
                    st.subheader("Feature Importance")
                    feature_names = num_features + cat_features
                    fig = plot_feature_importance(model, feature_names)
                    if fig:
                        st.pyplot(fig)
                    st.subheader("SHAP Values")
                    fig = plot_shap_values(model, X, feature_names)
                    if fig:
                        st.pyplot(fig)
                    st.subheader("Prediction Explanation")
                    example_idx = st.slider("Select an example to explain", 0, len(X)-1, 0)
                    if isinstance(model, Pipeline):
                        transformer_steps = model.steps[:-1]
                        transformer = Pipeline(transformer_steps)
                        X_transformed = transformer.transform(X.iloc[[example_idx]])
                    else:
                        X_transformed = X.iloc[[example_idx]]
                    st.write("Selected example features:")
                    st.dataframe(X.iloc[[example_idx]])
                    st.write(f"Predicted probability of default: {y_pred_proba[example_idx]:.2f}")
                    st.write(f"Actual outcome: {'Default' if y.iloc[example_idx] == 1 else 'Non-Default'}")
                except FileNotFoundError:
                    st.error(f"Model file '{model_path}' not found. Please ensure the model file is in the correct directory.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
