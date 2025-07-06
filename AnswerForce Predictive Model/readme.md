Overview
This guide provides setup instructions for the AnswerForce customer conversion prediction project, cloned from GitHub. The repository is https://github.com/janpashashaik123/AnswerForce.git and contains model.py, inference.py, requirements.txt, a dataset folder, and AnswerForce_Chatbot_Analysis.ipynb for Tasks 1 and 2.
Prerequisites

Clone the github repository:
run : git clone  https://github.com/janpashashaik123/AnswerForce.git

After clone, you should see a folder named AnswerForce Predictive Model.
Navigate into the folder:cd "AnswerForce Predictive Model"


Verify files: model.py, inference.py, requirements.txt, dataset/, AnswerForce_Chatbot_Analysis.ipynb should be present.

Install Dependencies

In the terminal, within the AnswerForce Predictive Model folder, run:pip install -r requirements.txt


Expected packages:
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
imbalanced-learn>=0.8.0
joblib>=1.0.0


Prepare Dataset

Ensure the dataset folder contains:
account_data.csv: Columns like Acct id, Acct type, Activate chat bot, Converted to paid customer.
usage_data.csv: Columns like Acct id, Date time, Number of clicks.



Run Scripts
Train the Model (model.py)

Trains a Random Forest Classifier with feature engineering and SMOTE.
Run:python model.py


Expected Output:
Trained model saved (e.g., random_forest_model.pkl).
Visualizations saved (confusion_matrix.png, loss_curve.png, image1.png).
Console output with accuracy (e.g., 74.73%), precision, recall, F1.


Perform Inference (inference.py)

Run:python inference.py 


Expected Output:
Conversion Probability for the sample: predicted probability


Optional Parameters: If hardcoded, use python inference.py.

# Analyze Data for Tasks 1 and 2 (AnswerForce_Chatbot_Analysis.ipynb)

Performs initial data exploration and visualization.
Run:python AnswerForce_Chatbot_Analysis.ipynb


Expected Output:
Console output with insights (e.g., conversion rates: ENT 51%, Active Chatbot ~41%).
Saved plots: image1.png (histogram), image2.png (time series).



Notes

Customization: Update ZIP file name or folder path if different.
Timeliness: Start actions (e.g., data validation, model training) by July 07, 2025, as per project timelines.
Troubleshooting:
Extraction Error: Verify ZIP file integrity or use a different extractor.
Dependency Issue: Re-run pip install -r requirements.txt.
Script Failure: Check dataset files or share error logs.


Support: Contact for assistance if needed.
