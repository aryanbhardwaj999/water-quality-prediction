
import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Water Potability Predictor", layout="centered")

# Safe ranges (based on WHO standards)
SAFE_RANGES = {
    'ph': (6.5, 8.5),
    'Hardness': (50, 150),
    'Solids': (200, 400),
    'Chloramines': (1, 3),
    'Sulfate': (200, 400),
    'Conductivity': (200, 800),
    'Organic_carbon': (1, 10),
    'Trihalomethanes': (0, 80),
    'Turbidity': (0, 1)
}

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_model()
scaler = load_scaler()


st.title("💧 Water Quality Prediction")
st.write("Enter water parameters to check if it's **Potable** or **Not Potable**.")

def make_prediction():
    user_input = st.session_state["form_values"]

    is_safe = all(SAFE_RANGES[k][0] <= user_input[k] <= SAFE_RANGES[k][1] for k in user_input)

    feature_order = ['ph', 'Hardness', 'Solids', 'Chloramines',
                     'Sulfate', 'Conductivity', 'Organic_carbon',
                     'Trihalomethanes', 'Turbidity']
    features = [user_input[k] for k in feature_order]

    # ✅ If everything is safe, skip model
    if is_safe:
        result = "✅ Potable"
        confidence = 1.0
    else:
        # ❌ Use model only when outside safe limits
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)[0]
        confidence = model.predict_proba(scaled_features)[0][prediction]
        result = "✅ Potable" if prediction == 1 else "❌ Not Potable"

    st.session_state["result"] = result
    st.session_state["confidence"] = confidence
    st.session_state["safe_check"] = is_safe
    st.session_state["features"] = features
    st.session_state["input_values"] = user_input


# Input form
with st.form("prediction_form"):
    form_values = {}
    for param, (min_val, max_val) in SAFE_RANGES.items():
        form_values[param] = st.number_input(
            f"{param.replace('_', ' ').title()} ({min_val}–{max_val})",
            min_value=0.0,
            step=0.01,
            key=param
        )

    submit_button = st.form_submit_button("Predict Potability")
    if submit_button:
        st.session_state["form_values"] = form_values
        make_prediction()

# Display result
if "result" in st.session_state:
    st.subheader("🔍 Prediction Result")
    st.success(f"**Result:** {st.session_state['result']}")
    st.write(f"**Model Confidence:** {st.session_state['confidence'] * 100:.2f}%")

    st.markdown("---")
    st.subheader("🧪 Parameter Analysis")
    for param in ['ph', 'Hardness', 'Solids', 'Chloramines',
                  'Sulfate', 'Conductivity', 'Organic_carbon',
                  'Trihalomethanes', 'Turbidity']:
        val = st.session_state["input_values"][param]
        safe = "✅ Safe" if SAFE_RANGES[param][0] <= val <= SAFE_RANGES[param][1] else "⚠️ Unsafe"
        st.write(f"**{param.replace('_', ' ').title()}**: {val} ({safe})")
