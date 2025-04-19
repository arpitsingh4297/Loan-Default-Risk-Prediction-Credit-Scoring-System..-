import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import requests
import io
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

# --- Load Models from Google Drive ---
def download_file_from_google_drive(id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

@st.cache_resource()
def load_model_from_drive(file_id, model_name):
    try:
        # Try to load from local cache first
        return joblib.load(model_name)
    except:
        try:
            # Download from Google Drive if not found locally
            download_file_from_google_drive(file_id, model_name)
            return joblib.load(model_name)
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            return None

# Model file IDs from Google Drive
best_credit_risk_id = "15f8MTVbecl955ElY_OiFQMCzodZ95GpH"
random_forest_id = "10Ez1zmYeI1gJv3n07ciXgRUR_l_3dROr"

# --- Load Data ---
@st.cache_data()
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
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
            
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                st.warning("Missing values detected:")
                st.dataframe(missing_values[missing_values > 0])
                
                # Basic imputation
                if 'Rate' in df.columns:
                    df['Rate'] = df.groupby('Intent')['Rate'].transform(lambda x: x.fillna(x.median()))
                
                # Create temporary income quartile for employment length imputation
                if 'Income' in df.columns and 'Emp_length' in df.columns:
                    df['Income_quartile'] = pd.qcut(df['Income'], q=4, labels=False)
                    df['Emp_length'] = df.groupby('Income_quartile')['Emp_length'].transform(lambda x: x.fillna(x.median()))
                    df.drop('Income_quartile', axis=1, inplace=True)
            else:
                st.success("No missing values found")
            
            st.write(f"Duplicate rows: {df.duplicated().sum()}")
            
            # Clean target variable
            if 'Default' in df.columns:
                df['Default'] = df['Default'].astype(str).str.strip().str.lower().map({
                    'y': 1, 'yes': 1, '1': 1,
                    'n': 0, 'no': 0, '0': 0
                }).fillna(0)
            
            # Ensure all expected columns are present
            expected_cols = list(data_dict.keys())
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                st.warning(f"Missing expected columns: {missing_cols}")
            
            return df, data_dict
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None, None
    return None, None

def create_features(df):
    df = df.copy()
    
    # Create derived features
    if 'Amount' in df.columns and 'Income' in df.columns:
        df['debt_to_income'] = df['Amount'] / df['Income']
        df['loan_to_income'] = df['Amount'] / df['Income']
    
    if 'Age' in df.columns:
        df['age_bucket'] = pd.cut(df['Age'], bins=[0, 30, 50, 100], 
                                 labels=['Young', 'Middle-aged', 'Senior'], 
                                 include_lowest=True)
    
    if 'Rate' in df.columns and 'Percent_income' in df.columns:
        df['rate_income_interaction'] = df['Rate'] * df['Percent_income']
    
    return df

def predict(model, data):
    try:
        # Ensure we're working with a DataFrame
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
            
        # Handle potential missing columns
        if hasattr(model, 'feature_names_in_'):
            missing_cols = set(model.feature_names_in_) - set(data.columns)
            if missing_cols:
                for col in missing_cols:
                    data[col] = 0  # Add missing columns with default value
        
        y_pred = model.predict(data)
        y_pred_proba = model.predict_proba(data)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Convert string predictions to numeric if needed
        if isinstance(y_pred[0], str):
            y_pred = pd.Series(y_pred).str.strip().str.lower().map({
                'y': 1, 'yes': 1, 'n': 0, 'no': 0, '0': 0, '1': 1
            }).values
        
        return y_pred, y_pred_proba
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# [Rest of your plotting and visualization functions remain the same...]

def main():
    st.title("Loan Default Prediction Dashboard")
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the section", 
                                   ["Data Overview", "Exploratory Analysis", "Model Prediction"])
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
    
    df, data_dict = load_data(uploaded_file)
    
    if df is not None:
        if app_mode == "Data Overview":
            st.header("Data Overview")
            st.write("First 5 rows of the dataset:")
            st.dataframe(df.head())
            st.subheader("Data Dictionary")
            st.dataframe(pd.DataFrame.from_dict(data_dict, orient='index', columns=['Description']))
            
        elif app_mode == "Exploratory Analysis":
            show_exploratory_analysis(df)
            
        elif app_mode == "Model Prediction":
            try:
                # Prepare data - ensure we don't modify the original df
                X_full = df.drop(columns=['Default'], errors='ignore')
                
                # Create features - handle potential missing columns gracefully
                X = create_features(X_full.drop(columns=['Id'], errors='ignore'))
                
                if 'Default' in df.columns:
                    y = df['Default']
                else:
                    y = None
                
                model_choice = st.selectbox("Select a Model", 
                                          ["Logistic Regression", "Random Forest"])
                
                if model_choice == "Logistic Regression":
                    model = load_model_from_drive(best_credit_risk_id, "best_credit_risk_model.pkl")
                elif model_choice == "Random Forest":
                    model = load_model_from_drive(random_forest_id, "random_forest_model.pkl")
                
                if model is not None:
                    st.subheader("Model Predictions")
                    
                    # Make predictions
                    y_pred, y_pred_proba = predict(model, X)
                    
                    if y_pred is not None:
                        if y is not None:
                            show_metrics(y, y_pred, y_pred_proba, model_choice)
                            
                            st.subheader("Model Performance")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.pyplot(plot_roc_curve(y, y_pred_proba, model_choice))
                            with col2:
                                st.pyplot(plot_precision_recall_curve(y, y_pred_proba, model_choice))
                        
                        st.subheader("Feature Importance")
                        feature_names = X.columns.tolist()
                        fig = plot_feature_importance(model, feature_names)
                        if fig:
                            st.pyplot(fig)
                        
                        st.subheader("SHAP Values")
                        fig = plot_shap_values(model, X, feature_names)
                        if fig:
                            st.pyplot(fig)
                        
                        st.subheader("Prediction Explanation")
                        example_idx = st.slider("Select an example to explain", 0, len(X)-1, 0)
                        st.write("Selected example features:")
                        st.dataframe(X.iloc[[example_idx]])
                        
                        if y_pred_proba is not None:
                            st.write(f"Predicted probability of default: {y_pred_proba[example_idx]:.2f}")
                        
                        if y is not None:
                            st.write(f"Actual outcome: {'Default' if y.iloc[example_idx] == 1 else 'Non-Default'}")
            except Exception as e:
                st.error(f"Error in Model Prediction: {str(e)}")

if __name__ == "__main__":
    main()
