import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
import os

# --- Constants ---
HF_MODEL_REPO_ID = "ShaksML/dt-model-predective-maintenance"
HF_MODEL_FILENAME = "optimized_dt_predictor.pkl"

# --- Function to Load Model from Hugging Face ---
@st.cache_resource
def load_model():
    """Downloads the model artifact from the Hugging Face Hub and loads it."""
    try:
        model_path = hf_hub_download(
            repo_id=HF_MODEL_REPO_ID,
            filename=HF_MODEL_FILENAME,
            repo_type="model",
            local_dir=".",
            local_dir_use_symlinks=False
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"System Error: Could not retrieve model. {e}")
        st.stop()

# --- Streamlit Application Layout ---
st.set_page_config(page_title="Engine Health Monitor", layout="wide", page_icon="⚙️")

# Load the trained model
model = load_model()

# Input columns (Original names as seen in the form)
INPUT_COLUMNS = [
    'Engine rpm', 'Lub oil pressure', 'Fuel pressure',
    'Coolant pressure', 'lub oil temp', 'Coolant temp'
]

# --- Sidebar UI for Inputs ---
st.sidebar.header("⚙️ Sensor Configuration")
st.sidebar.info("Adjust the parameters below to simulate engine conditions.")

input_data = {}

# Grouping inputs in the sidebar for a cleaner look
with st.sidebar.form("sensor_inputs_form"):
    st.subheader("Pressure Readings")
    input_data['Lub oil pressure'] = st.slider("Lub Oil Pressure (bar)", 0.0, 8.0, 3.30, 0.01)
    input_data['Fuel pressure'] = st.slider("Fuel Pressure (bar)", 0.0, 25.0, 6.60, 0.01)
    input_data['Coolant pressure'] = st.slider("Coolant Pressure (bar)", 0.0, 8.0, 2.30, 0.01)

    st.subheader("Temperature & RPM")
    input_data['Engine rpm'] = st.number_input("Engine RPM", 400, 2000, 890, 10)
    input_data['lub oil temp'] = st.number_input("Lub Oil Temp (°C)", 70.0, 100.0, 78.0, 0.1)
    input_data['Coolant temp'] = st.number_input("Coolant Temp (°C)", 70.0, 110.0, 78.0, 0.1)

    # Submit Button
    submit_button = st.form_submit_button("Run Diagnostics")

# --- Main Page UI ---
st.title("Predictive Maintenance System")
st.markdown("### 📊 Live Engine Analysis")

if model is not None:
    if submit_button:
        # 1. Create DataFrame with input data
        input_df = pd.DataFrame([input_data])

        # 2. Organize columns to match original input structure
        input_df = input_df[INPUT_COLUMNS]

        # 3. RENAME COLUMNS TO MATCH MODEL TRAINING DATA (*** CRITICAL FIX ***)
        rename_map = {
            'Engine rpm': 'Engine_RPM',
            'Lub oil pressure': 'Lub_Oil_Pressure',
            'Fuel pressure': 'Fuel_Pressure',
            'Coolant pressure': 'Coolant_Pressure',
            'lub oil temp': 'Lub_Oil_Temperature',
            'Coolant temp': 'Coolant_Temperature'
        }
        input_df.rename(columns=rename_map, inplace=True)

        # 4. Make Prediction
        try:
            prediction_proba = model.predict_proba(input_df)[0]
            prediction = model.predict(input_df)[0]

            st.divider()
            
            # Use columns for a dashboard-like result display
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Diagnostic Report")
                if prediction == 1:
                    st.error("⚠️ **CRITICAL ALERT: Engine Fault Detected**")
                    st.markdown("The system analysis indicates a high probability of component failure. Immediate inspection is recommended.")
                else:
                    st.success("**STATUS: Optimal**")
                    st.markdown("All systems are functioning within normal operational parameters.")

            with col2:
                st.subheader("Confidence Level")
                if prediction == 1:
                    st.metric(label="Failure Probability", value=f"{prediction_proba[1]*100:.2f}%", delta="High Risk", delta_color="inverse")
                else:
                    st.metric(label="Health Score", value=f"{prediction_proba[0]*100:.2f}%", delta="Stable")

            # Expandable section for raw data
            with st.expander("View Processed Sensor Telemetry"):
                st.dataframe(input_df, hide_index=True, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during prediction processing: {e}")
    else:
        st.info("👈 Please configure sensors in the sidebar and click 'Run Diagnostics' to begin.")
else:
    st.warning("Model definition not found.")
