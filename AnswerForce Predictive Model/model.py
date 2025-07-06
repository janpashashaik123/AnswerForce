import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib

def load_data(attributes_file="/content/account_attributes.xlsx", usage_file="/content/account_usage.xlsx"):
    """Load and validate datasets."""
    try:
        df_attributes = pd.read_excel(attributes_file)
        df_usage = pd.read_excel(usage_file)

        required_attr_cols = ['Acct id', 'Acct type', 'Activate chat bot', 'Converted to paid customer']
        required_usage_cols = ['Acct id', 'Date time', 'Number of link clicks']
        if not all(col in df_attributes.columns for col in required_attr_cols):
            raise ValueError("Missing required columns in account_attributes.xlsx: " + str(required_attr_cols))
        if not all(col in df_usage.columns for col in required_usage_cols):
            raise ValueError("Missing required columns in account_usage.xlsx: " + str(required_usage_cols))

        df_usage = df_usage.rename(columns={"Number of link clicks": "Number of clicks"})
        return df_attributes, df_usage
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def preprocess_data(df_attributes, df_usage):
    """Handle missing values and validate data formats."""
    try:
        df_attributes = df_attributes.fillna({'Acct type': 'SMB', 'Activate chat bot': 'N', 'Converted to paid customer': 0})
        df_usage = df_usage.dropna(subset=['Number of clicks', 'Date time'])
        df_usage['Date time'] = pd.to_datetime(df_usage['Date time'], errors='coerce')
        if df_usage['Date time'].isnull().any():
            raise ValueError("Invalid date formats in usage data")
        return df_attributes, df_usage
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        raise

def engineer_features(df_attributes, df_usage):
    """Aggregate usage data and create new features."""
    try:
        usage_agg = df_usage.groupby('Acct id').agg({
            'Number of clicks': ['sum', 'mean', 'count'],
            'Date time': ['min', 'max']
        }).reset_index()
        usage_agg.columns = ['Acct id', 'total_clicks', 'avg_weekly_clicks', 'num_weeks_active', 'first_active_date', 'last_active_date']
        usage_agg['trial_duration_days'] = (usage_agg['last_active_date'] - usage_agg['first_active_date']).dt.days

        df_usage['Week'] = df_usage['Date time'].dt.isocalendar().week
        weekly_clicks = df_usage.groupby(['Acct id', 'Week'])['Number of clicks'].sum().reset_index()
        click_variability = weekly_clicks.groupby('Acct id')['Number of clicks'].std().reset_index()
        click_variability.columns = ['Acct id', 'click_variability']
        usage_agg = pd.merge(usage_agg, click_variability, on='Acct id', how='left')
        usage_agg['click_variability'] = usage_agg['click_variability'].fillna(0)

        df_merged = pd.merge(df_attributes, usage_agg, on='Acct id', how='left')
        df_merged = df_merged.fillna({'total_clicks': 0, 'avg_weekly_clicks': 0, 'num_weeks_active': 0, 'trial_duration_days': 0, 'click_variability': 0})

        le_acct_type = LabelEncoder()
        le_chatbot = LabelEncoder()
        df_merged['Acct type'] = le_acct_type.fit_transform(df_merged['Acct type'])
        df_merged['Activate chat bot'] = le_chatbot.fit_transform(df_merged['Activate chat bot'])

        df_merged['clicks_chatbot_interaction'] = df_merged['total_clicks'] * df_merged['Activate chat bot']

        features = ['Acct type', 'Activate chat bot', 'total_clicks', 'avg_weekly_clicks', 'num_weeks_active', 'trial_duration_days', 'click_variability', 'clicks_chatbot_interaction']
        missing_cols = [col for col in features if col not in df_merged.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns in df_merged: {missing_cols}")

        X = df_merged[features]
        y = df_merged['Converted to paid customer']
        return X, y, le_acct_type, le_chatbot
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        raise

def scale_features(X):
    """Scale numerical features."""
    try:
        scaler = StandardScaler()
        numerical_features = ['total_clicks', 'avg_weekly_clicks', 'num_weeks_active', 'trial_duration_days', 'click_variability', 'clicks_chatbot_interaction']
        missing_num_cols = [col for col in numerical_features if col not in X.columns]
        if missing_num_cols:
            raise ValueError(f"Missing numerical feature columns in X: {missing_num_cols}")
        X[numerical_features] = scaler.fit_transform(X[numerical_features])
        return X, scaler
    except Exception as e:
        print(f"Error in scaling features: {e}")
        raise

def balance_data(X, y):
    """Handle class imbalance with SMOTE."""
    try:
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
        return X, y
    except Exception as e:
        print(f"Error in balancing data: {e}")
        raise

def train_random_forest(X, y):
    """Train Random Forest model with fixed hyperparameters and compute OOB errors."""
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Compute OOB errors for loss curve
        oob_errors = []
        tree_counts = list(range(30, 201, 10))  # Start at 30 to avoid OOB warning
        for n_trees in tree_counts:
            rf_temp = RandomForestClassifier(
                max_depth=20,
                min_samples_leaf=1,
                min_samples_split=2,
                n_estimators=n_trees,
                random_state=42,
                oob_score=True
            )
            rf_temp.fit(X_train, y_train)
            oob_errors.append(1 - rf_temp.oob_score_)

        # Train final model
        rf_model = RandomForestClassifier(
            max_depth=20,
            min_samples_leaf=1,
            min_samples_split=2,
            n_estimators=200,
            random_state=42,
            oob_score=True
        )
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Compute confusion matrix and metrics
        cm = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        return rf_model, accuracy, feature_importance, X_test, y_test, cm, precision, recall, f1, oob_errors, tree_counts
    except Exception as e:
        print(f"Error in training Random Forest: {e}")
        raise

def predict_samples(model, scaler, le_acct_type, le_chatbot, samples):
    """Predict conversion probabilities for a list of samples."""
    try:
        features = ['Acct type', 'Activate chat bot', 'total_clicks', 'avg_weekly_clicks', 'num_weeks_active', 'trial_duration_days', 'click_variability', 'clicks_chatbot_interaction']

        df_samples = pd.DataFrame(samples)

        missing_cols = [col for col in features if col not in df_samples.columns]
        if missing_cols:
            raise ValueError(f"Missing features in samples: {missing_cols}")

        df_samples['Acct type'] = le_acct_type.transform(df_samples['Acct type'])
        df_samples['Activate chat bot'] = le_chatbot.transform(df_samples['Activate chat bot'])

        numerical_features = ['total_clicks', 'avg_weekly_clicks', 'num_weeks_active', 'trial_duration_days', 'click_variability', 'clicks_chatbot_interaction']
        df_samples[numerical_features] = scaler.transform(df_samples[numerical_features])

        probabilities = model.predict_proba(df_samples[features])[:, 1]
        predictions = pd.DataFrame({
            'Sample Index': df_samples.index,
            'Conversion Probability': probabilities
        })
        return predictions
    except Exception as e:
        print(f"Error in predicting samples: {e}")
        raise

def create_visualizations(accuracy, cm, precision, recall, f1, oob_errors, tree_counts):
    """Create and save loss curve and confusion matrix visualizations."""
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(tree_counts, oob_errors, marker='o')
        plt.title('Random Forest Training Loss (OOB Error) vs. Number of Trees')
        plt.xlabel('Number of Trees')
        plt.ylabel('Out-of-Bag Error')
        plt.savefig('loss_curve.png')
        plt.close()

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('confusion_matrix.png')
        plt.close()
    except Exception as e:
        print(f"Error in creating visualizations: {e}")
        raise

def save_model_and_performance(model, accuracy, cm, precision, recall, f1, feature_importance):
    """Save model weights and print performance metrics."""
    try:
        joblib.dump(model, 'model.joblib')
        performance_text = f"""
### Random Forest Model Performance
Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1 Score: {f1:.4f}
Confusion Matrix:
{cm}
Feature Importance:
{feature_importance.to_string(index=False)}
"""
        print(performance_text)
    except Exception as e:
        print(f"Error in saving model/performance: {e}")
        raise

def save_preprocessors(scaler, le_acct_type, le_chatbot):
    """Save preprocessors for inference."""
    try:
        joblib.dump(scaler, 'scaler.joblib')
        joblib.dump(le_acct_type, 'le_acct_type.joblib')
        joblib.dump(le_chatbot, 'le_chatbot.joblib')
    except Exception as e:
        print(f"Error in saving preprocessors: {e}")
        raise

def create_model(attributes_file="/content/account_attributes.xlsx", usage_file="/content/account_usage.xlsx"):
    """Orchestrate training: Train Random Forest, evaluate, and predict samples."""
    try:
        df_attributes, df_usage = load_data(attributes_file, usage_file)
        df_attributes, df_usage = preprocess_data(df_attributes, df_usage)
        X, y, le_acct_type, le_chatbot = engineer_features(df_attributes, df_usage)
        X, scaler = scale_features(X)
        X, y = balance_data(X, y)
        model, accuracy, feature_importance, X_test, y_test, cm, precision, recall, f1, oob_errors, tree_counts = train_random_forest(X, y)
        create_visualizations(accuracy, cm, precision, recall, f1, oob_errors, tree_counts)
        save_model_and_performance(model, accuracy, cm, precision, recall, f1, feature_importance)
        save_preprocessors(scaler, le_acct_type, le_chatbot)

        sample = [{
            'Acct type': 'ENT',
            'Activate chat bot': 'Y',
            'total_clicks': 10000,
            'avg_weekly_clicks': 500,
            'num_weeks_active': 20,
            'trial_duration_days': 140,
            'click_variability': 50,
            'clicks_chatbot_interaction': 10000
        }]
        predictions = predict_samples(model, scaler, le_acct_type, le_chatbot, sample)
        print("\nSample Predictions:")
        print(predictions.to_string(index=False))

        print(f"\nTraining completed successfully. Accuracy: {accuracy:.4f}.")
        return {'Random Forest': accuracy}
    except Exception as e:
        print(f"Error in training: {e}")
        raise

if __name__ == "__main__":
    attributes_file = "dataset/account_attributes.xlsx"
    usage_file = "dataset/account_usage.xlsx"
    accuracies = create_model(attributes_file=attributes_file, usage_file=usage_file)
    print("Model Accuracy:")
    for name, acc in accuracies.items():
        print(f"- {name}: {acc:.4f}")