import pandas as pd
import joblib

def predict_single_sample(sample, model_path='model.joblib', scaler_path='scaler.joblib', le_acct_type_path='le_acct_type.joblib', le_chatbot_path='le_chatbot.joblib'):
    """Predict conversion probability for a single sample."""
    try:
        features = ['Acct type', 'Activate chat bot', 'total_clicks', 'avg_weekly_clicks', 'num_weeks_active', 'trial_duration_days', 'click_variability', 'clicks_chatbot_interaction']

        df_sample = pd.DataFrame([sample])

        missing_cols = [col for col in features if col not in df_sample.columns]
        if missing_cols:
            raise ValueError(f"Missing features in sample: {missing_cols}")

        scaler = joblib.load(scaler_path)
        le_acct_type = joblib.load(le_acct_type_path)
        le_chatbot = joblib.load(le_chatbot_path)

        df_sample['Acct type'] = le_acct_type.transform(df_sample['Acct type'])
        df_sample['Activate chat bot'] = le_chatbot.transform(df_sample['Activate chat bot'])

        numerical_features = ['total_clicks', 'avg_weekly_clicks', 'num_weeks_active', 'trial_duration_days', 'click_variability', 'clicks_chatbot_interaction']
        df_sample[numerical_features] = scaler.transform(df_sample[numerical_features])

        model = joblib.load(model_path)
        probability = model.predict_proba(df_sample[features])[:, 1][0]
        return probability
    except Exception as e:
        print(f"Error in predicting sample: {e}")
        raise

if __name__ == "__main__":
    sample = {
        'Acct type': 'ENT',
        'Activate chat bot': 'Y',
        'total_clicks': 10000,
        'avg_weekly_clicks': 500,
        'num_weeks_active': 20,
        'trial_duration_days': 140,
        'click_variability': 50,
        'clicks_chatbot_interaction': 10000
    }
    probability = predict_single_sample(sample)
    print(f"Conversion Probability for the sample: {probability:.4f}")