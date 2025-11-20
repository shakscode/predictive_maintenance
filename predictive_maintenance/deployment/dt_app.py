import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
import os

# --- Constants: MODIFIED FOR DECISION TREE MODEL (CODE 4) ---
# Use the exact repository ID from Code 4
HF_MODEL_REPO_ID = "ShaksML/dt-model-predective-maintenance"
# The model file was saved and uploaded as 'optimized_dt_predictor.pkl' in Code 4
HF_MODEL_FILENAME = "optimized_dt_predictor.pkl"
# Note: Since Code 4 trained a scikit-learn Pipeline (Scaler + DT), the pipeline handles preprocessing.

# --- Function to Load Model from Hugging Face ---
@st.cache_resource
def load_model():
    """Downloads the model artifact from the Hugging Face Hub and loads it."""
    try:
        # Download the model file from the Hugging Face repository
        model_path = hf_hub_download(
            repo_id=HF_MODEL_REPO_ID,
            filename=HF_MODEL_FILENAME,
            repo_type="model", # Explicitly set repo_type to 'model'
            local_dir=".",
            local_dir_use_symlinks=False
        )
        st.success(f"Model '{HF_MODEL_FILENAME}' successfully loaded from {HF_MODEL_REPO_ID}!")
        # Load the model using joblib (which is the format used by Code 4)
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from Hugging Face Hub. Please check repo ID and filename: {e}")
        st.stop() # Stop execution if the model cannot be loaded

# --- Streamlit Application Layout ---

st.set_page_config(
    page_title="DT Predictive Maintenance App",
    layout="wide"
)

st.title("Decision Tree Engine Maintenance Dashboard")
st.subheader("Forecast potential engine failures using real-time sensor data.")

# Load the trained model
model = load_model()

# Define the columns exactly as expected by the model (using the names from the original data)
# Assuming the column names are:
INPUT_COLUMNS = [
    'Engine rpm', 'Lub oil pressure', 'Fuel pressure',
    'Coolant pressure', 'lub oil temp', 'Coolant temp'
]

if model is not None:
    # --- Input Form for Sensor Readings ---
    st.markdown("---")
    st.header("Enter Engine Sensor Readings")

    # Dictionary to hold the user inputs
    input_data = {}

    # Define the input columns in a three-column layout
    col1, col2, col3 = st.columns(3)

    with col1:
        # Engine rpm (int)
        input_data['Engine rpm'] = st.number_input(
            "Engine RPM (Revolutions per Minute)",
            min_value=400, max_value=2000, value=790, step=10, key='rpm'
        )
        # Lub oil pressure (float)
        input_data['Lub oil pressure'] = st.number_input(
            "Lub Oil Pressure (bar)",
            min_value=0.0, max_value=8.0, value=3.30, step=0.1, format="%.2f", key='lop'
        )

    with col2:
        # Fuel pressure (float)
        input_data['Fuel pressure'] = st.number_input(
            "Fuel Pressure (bar)",
            min_value=0.0, max_value=25.0, value=6.60, step=0.1, format="%.2f", key='fp'
        )
        # Coolant pressure (float)
        input_data['Coolant pressure'] = st.number_input(
            "Coolant Pressure (bar)",
            min_value=0.0, max_value=8.0, value=2.30, step=0.1, format="%.2f", key='cp'
        )

    with col3:
        # lub oil temp (float)
        input_data['lub oil temp'] = st.number_input(
            "Lub Oil Temperature (°C)",
            min_value=70.0, max_value=100.0, value=78.0, step=0.1, format="%.2f", key='lot'
        )
        # Coolant temp (float)
        input_data['Coolant temp'] = st.number_input(
            "Coolant Temperature (°C)",
            min_value=70.0, max_value=110.0, value=78.0, step=0.1, format="%.2f", key='ct'
        )

    # --- Prediction Logic ---
    if st.button("Predict Engine Condition", type="primary"):
        # 1. Get the inputs and save them into a dataframe
        input_df = pd.DataFrame([input_data])

        # 2. Ensure the order of columns matches the training data (CRITICAL)
        input_df = input_df[INPUT_COLUMNS]

        # 3. Make Prediction
        try:
            # Predict probability for both classes (0 and 1)
            prediction_proba = model.predict_proba(input_df)[0]
            # Prediction is the class index (0 or 1)
            prediction = model.predict(input_df)[0]

            # 4. Display Result
            st.markdown("---")
            st.header("Prediction Result")

            if prediction == 1:
                st.error("🚨 FAULT PREDICTED (Requires Maintenance)")
                st.markdown(f"**Probability of Failure (Class 1):** `{prediction_proba[1]*100:.2f}%`")
                st.markdown("Immediate inspection and preventive maintenance are **strongly recommended** to avoid unexpected breakdown and costly repairs.")
            else:
                st.success("✅ NORMAL OPERATION")
                st.markdown(f"**Probability of Normal Operation (Class 0):** `{prediction_proba[0]*100:.2f}%`")
                st.markdown("The engine is operating within normal parameters. Continue with scheduled monitoring.")

            st.markdown("---")
            st.caption("Input Data Used for Prediction:")
            st.dataframe(input_df, hide_index=True) # Show the data that was fed to the model

        except Exception as e:
            st.error(f"An error occurred during prediction. Please check input values. Full error: {e}")

else:
    st.warning("Cannot proceed without a successfully loaded model. Please ensure the Decision Tree model exists in the Hugging Face repo.")
