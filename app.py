import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

st.title("Predictive Maintenance for Industrial Machinery")

# Show sample data
st.subheader("Download Sample CSV File To Train Model")
with open('data/machine_data.csv', 'rb') as my_file:
    st.download_button(label='Download CSV', data=my_file, file_name='sample_machine_data.csv', mime='text/csv')

# Upload CSV
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data", data.head())

    # Data Visualization
    st.subheader("Data Visualization")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    sns.histplot(data['Temperature'], ax=axs[0], kde=True)
    sns.histplot(data['Vibration'], ax=axs[1], kde=True)
    sns.histplot(data['Pressure'], ax=axs[2], kde=True)
    st.pyplot(fig)

    # Data Preprocessing
    X = data[['Temperature', 'Vibration', 'Pressure']]
    y = data['MaintenanceNeeded']

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Models
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Support Vector Machine": SVC(random_state=42),
        "XGBoost": xgb.XGBClassifier(random_state=42)
    }

    # Hyperparameter Tuning
    param_grids = {
        "Random Forest": {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        "Gradient Boosting": {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0]
        },
        "Logistic Regression": {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs']
        },
        "Support Vector Machine": {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        },
        "XGBoost": {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0]
        }
    }

    st.subheader("Model Evaluation")
    best_model = None
    best_accuracy = 0
    model_reports = {}

    for name, model in models.items():
        param_grid = param_grids[name]
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        model_reports[name] = classification_report(y_test, y_pred, output_dict=True)
        st.text(f"{name} Model (Best Params: {grid_search.best_params_})")
        st.text(classification_report(y_test, y_pred))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = grid_search.best_estimator_

    st.subheader(f"Best Model: {best_model.__class__.__name__} with Accuracy: {best_accuracy}")

    # Prediction
    st.subheader("Make Predictions")
    new_data = st.file_uploader("Upload New Data for Prediction", type="csv")
    if new_data:
        new_data_df = pd.read_csv(new_data)
        st.write("New Data", new_data_df.head())
        new_data_scaled = scaler.transform(new_data_df[['Temperature', 'Vibration', 'Pressure']])
        predictions = best_model.predict(new_data_scaled)
        new_data_df['PredictedMaintenance'] = predictions
        st.write("Predictions", new_data_df)

        # Download Predictions
        csv = new_data_df.to_csv(index=False)
        st.download_button(label="Download Predictions", data=csv, file_name='predictions.csv', mime='text/csv')
        
