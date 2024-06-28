import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

st.title("Predictive Maintenance for Industrial Machinery")

# Show sample data
st.subheader("Download Sample CSV File To Train model")
with open('data/machine_data.csv', 'rb') as my_file:
    st.download_button(label = 'Download CSV', data = my_file, file_name = 'sample_machine_data.csv', mime = 'text/csv')      


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

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model Training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model Evaluation
    st.subheader("Model Evaluation")
    st.text(classification_report(y_test, y_pred))

    # Prediction
    st.subheader("Make Predictions")
    new_data = st.file_uploader("Upload New Data for Prediction", type="csv")
    if new_data:
        new_data_df = pd.read_csv(new_data)
        st.write("New Data", new_data_df.head())
        predictions = model.predict(new_data_df[['Temperature', 'Vibration', 'Pressure']])
        new_data_df['PredictedMaintenance'] = predictions
        st.write("Predictions", new_data_df)

        # Download Predictions
        csv = new_data_df.to_csv(index=False)
        st.download_button(label="Download Predictions", data=csv, file_name='predictions.csv', mime='text/csv')
