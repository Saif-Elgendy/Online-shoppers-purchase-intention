
import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Purchase Intention Predictor", page_icon="🛒", layout="centered")

MODEL_PATH = "artifacts/purchase_intention_pipeline.joblib"

st.title("🛒 Will This Visitor Buy?")
st.write("Estimate whether an e-commerce browsing session is likely to result in a purchase.")
st.caption("Educational prototype. Predictions are estimates, not guarantees.")

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Run the notebook first and save the trained pipeline to artifacts/purchase_intention_pipeline.joblib.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.subheader("Enter session details")
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        administrative = st.number_input("Administrative pages", min_value=0, max_value=100, value=0)
        administrative_duration = st.number_input("Administrative time (seconds)", min_value=0.0, value=0.0)
        informational = st.number_input("Informational pages", min_value=0, max_value=100, value=0)
        informational_duration = st.number_input("Informational time (seconds)", min_value=0.0, value=0.0)
        product_related = st.number_input("Product-related pages", min_value=0, max_value=500, value=5)
        product_related_duration = st.number_input("Product-related time (seconds)", min_value=0.0, value=300.0)
        bounce_rates = st.number_input("Bounce rate", min_value=0.0, max_value=1.0, value=0.02, step=0.01)
        exit_rates = st.number_input("Exit rate", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    with col2:
        page_values = st.number_input("Page values", min_value=0.0, value=0.0)
        special_day = st.number_input("Special day (0 to 1)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        month = st.selectbox("Month", ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        operating_systems = st.selectbox("Operating system", list(range(1, 9)), index=1)
        browser = st.selectbox("Browser", list(range(1, 14)), index=1)
        region = st.selectbox("Region", list(range(1, 10)), index=0)
        traffic_type = st.selectbox("Traffic type", list(range(1, 21)), index=1)
        visitor_type = st.selectbox("Visitor type", ["Returning_Visitor", "New_Visitor", "Other"])
        weekend = st.selectbox("Weekend?", [False, True])

    submitted = st.form_submit_button("Predict purchase intention", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "Administrative": administrative,
        "Administrative_Duration": administrative_duration,
        "Informational": informational,
        "Informational_Duration": informational_duration,
        "ProductRelated": product_related,
        "ProductRelated_Duration": product_related_duration,
        "BounceRates": bounce_rates,
        "ExitRates": exit_rates,
        "PageValues": page_values,
        "SpecialDay": special_day,
        "Month": month,
        "OperatingSystems": operating_systems,
        "Browser": browser,
        "Region": region,
        "TrafficType": traffic_type,
        "VisitorType": visitor_type,
        "Weekend": weekend
    }])

    try:
        probability = float(model.predict_proba(row)[0][1])
        st.subheader("Prediction")
        st.metric("Estimated purchase probability", f"{probability:.1%}")
        st.progress(probability)
        if probability >= 0.70:
            st.success("Likely to purchase")
            st.info("Suggested action: No discount is needed by default; avoid giving away margin unnecessarily.")
        elif probability >= 0.35:
            st.warning("Moderate purchase likelihood")
            st.info("Suggested action: Consider helpful product information or a chat assistant.")
        else:
            st.error("Low purchase likelihood")
            st.info("Suggested action: Consider a carefully targeted offer, subject to business rules.")
        st.caption("These action bands are illustrative business rules. Validate thresholds using real campaign results.")
    except Exception as exc:
        st.error(f"Could not generate prediction: {exc}")
