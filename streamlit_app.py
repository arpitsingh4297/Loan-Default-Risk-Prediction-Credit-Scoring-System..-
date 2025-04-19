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

def show_metrics(y_true, y_pred, y_pred_proba, model_name="Model"):
    try:
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
        
        if y_pred_proba is not None:
            st.write(f"ROC AUC: {roc_auc_score(y_true, y_pred_proba):.3f}")
            st.write(f"PR AUC: {average_precision_score(y_true, y_pred_proba):.3f}")
    except Exception as e:
        st.error(f"Error showing metrics: {str(e)}")

def plot_roc_curve(y_true, y_pred_proba, model_name="Model"):
    try:
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
    except Exception as e:
        st.error(f"Error plotting ROC curve: {str(e)}")
        return None

def plot_precision_recall_curve(y_true, y_pred_proba, model_name="Model"):
    try:
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
    except Exception as e:
        st.error(f"Error plotting precision-recall curve: {str(e)}")
        return None

def plot_feature_importance(model, feature_names):
    try:
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
        else:
            st.warning("Model doesn't have feature_importances_ attribute")
            return None
    except Exception as e:
        st.error(f"Error plotting feature importance: {str(e)}")
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

def show_exploratory_analysis(df):
    st.header("Exploratory Data Analysis")
    
    # Target variable distribution
    if 'Default' in df.columns:
        st.subheader("Target Variable Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='Default', data=df, ax=ax)
        ax.set_title('Distribution of Loan Defaults')
        ax.set_xlabel('Default Status')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    
    # Numerical features distribution
    st.subheader("Numerical Features Distribution")
    num_cols = ['Age', 'Income', 'Amount', 'Rate', 'Percent_income', 'Cred_length']
    available_num_cols = [col for col in num_cols if col in df.columns]
    
    if available_num_cols:
        selected_num = st.selectbox("Select numerical feature to visualize", available_num_cols)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(df[selected_num], kde=True, ax=ax1)
        ax1.set_title(f'Distribution of {selected_num}')
        
        if 'Default' in df.columns:
            sns.boxplot(x='Default', y=selected_num, data=df, ax=ax2)
            ax2.set_title(f'{selected_num} by Default Status')
        else:
            sns.boxplot(y=selected_num, data=df, ax=ax2)
            ax2.set_title(f'Boxplot of {selected_num}')
        
        st.pyplot(fig)
    else:
        st.warning("No numerical features available for visualization")
    
    # Categorical features distribution
    st.subheader("Categorical Features Distribution")
    cat_cols = ['Home', 'Intent', 'Status']
    available_cat_cols = [col for col in cat_cols if col in df.columns]
    
    if available_cat_cols:
        selected_cat = st.selectbox("Select categorical feature to visualize", available_cat_cols)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        value_counts = df[selected_cat].value_counts()
        ax1.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Distribution of {selected_cat}')
        
        if 'Default' in df.columns:
            sns.countplot(x=selected_cat, hue='Default', data=df, ax=ax2)
            ax2.set_title(f'Default Rate by {selected_cat}')
        else:
            sns.countplot(x=selected_cat, data=df, ax=ax2)
            ax2.set_title(f'Count of {selected_cat}')
        
        ax2.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
    else:
        st.warning("No categorical features available for visualization")
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    corr_cols = ['Age', 'Income', 'Amount', 'Rate', 'Percent_income', 'Cred_length', 'Default']
    available_corr_cols = [col for col in corr_cols if col in df.columns]
    
    if len(available_corr_cols) > 1:
        corr_matrix = df[available_corr_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlation Matrix')
        st.pyplot(fig)
    else:
        st.warning("Not enough numerical features for correlation analysis")

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
                    st.warning("No 'Default' column found - can't calculate metrics")
                
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
                                roc_fig = plot_roc_curve(y, y_pred_proba, model_choice)
                                if roc_fig:
                                    st.pyplot(roc_fig)
                            
                            with col2:
                                pr_fig = plot_precision_recall_curve(y, y_pred_proba, model_choice)
                                if pr_fig:
                                    st.pyplot(pr_fig)
                        
                        st.subheader("Feature Importance")
                        feature_names = X.columns.tolist()
                        fi_fig = plot_feature_importance(model, feature_names)
                        if fi_fig:
                            st.pyplot(fi_fig)
                        
                        st.subheader("SHAP Values")
                        shap_fig = plot_shap_values(model, X, feature_names)
                        if shap_fig:
                            st.pyplot(shap_fig)
                        
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
