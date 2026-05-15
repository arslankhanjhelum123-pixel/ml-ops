import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Credit Risk Predictor", layout="wide")
st.title("🏦 German Credit Risk Predictor")
st.markdown("### Predict Creditability (Good / Bad)")

# Load model
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        model_info = pickle.load(f)
    return model_info["model"]

model = load_model()

# Real column labels
labels = {
    'COL 1': "Account Balance",
    'COL 2': "Duration of Credit (month)",
    'COL 3': "Payment Status of Previous Credit",
    'COL 4': "Purpose",
    'COL 5': "Credit Amount",
    'COL 6': "Value Savings/Stocks",
    'COL 7': "Length of current employment",
    'COL 8': "Instalment percent",
    'COL 9': "Sex & Marital Status",
    'COL 10': "Guarantors",
    'COL 11': "Duration in Current address",
    'COL 12': "Most valuable available asset",
    'COL 13': "Age (years)",
    'COL 14': "Concurrent Credits",
    'COL 15': "Type of apartment",
    'COL 16': "No of Credits at this Bank",
    'COL 17': "Occupation",
    'COL 18': "No of dependents",
    'COL 19': "Telephone",
    'COL 20': "Foreign Worker"
}

st.sidebar.header("📋 Customer Information")

def user_input_features():
    data = {}
    col1, col2 = st.sidebar.columns(2)
    
    for i in range(1, 21):
        col_name = f'COL {i}'
        label = labels.get(col_name, col_name)
        
        with (col1 if i % 2 == 1 else col2):
            # Most fields are numeric
            if col_name in ['COL 2', 'COL 5', 'COL 8', 'COL 11', 'COL 13', 'COL 16', 'COL 18']:
                value = st.number_input(label, value=0, step=1, key=col_name)
            else:
                value = st.text_input(label, value="0", key=col_name)
            
            data[col_name] = value
    
    return pd.DataFrame([data])

input_df = user_input_features()

st.subheader("Input Summary")
st.dataframe(input_df)

if st.button("🔍 Predict Credit Risk", type="primary", use_container_width=True):
    with st.spinner("Making Prediction..."):
        try:
            # Convert to numeric
            input_numeric = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            prediction = model.predict(input_numeric)[0]
            proba = model.predict_proba(input_numeric)[0]

            if prediction == 1:
                st.success("✅ **GOOD CREDIT** - Low Risk Customer")
                st.balloons()
            else:
                st.error("⚠️ **BAD CREDIT** - High Risk Customer")

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Good Credit Probability", f"{proba.max()*100:.1f}%")
            with col_b:
                st.metric("Bad Credit Probability", f"{proba.min()*100:.1f}%")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

st.caption("Note: Enter values according to the actual meaning of each field.")