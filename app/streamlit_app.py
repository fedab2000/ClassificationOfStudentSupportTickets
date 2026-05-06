import streamlit as st
import joblib
import pandas as pd

# MUST be first Streamlit command
st.set_page_config(page_title="AI Ticket Router", layout="centered")


def load_model():
    # IMPORTANT: match this path with where your notebook saves the model
    return joblib.load("outputs/ticket_model.joblib")


def predict_with_explanation(model, text):
    probs = model.predict_proba([text])[0]
    classes = model.classes_
    return sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:3]


# Load model
model = load_model()

# UI
st.title("🎓 Higher Education AI Support Ticket Router")

st.write(
    "Enter a student support request. The model will predict the best department "
    "and show confidence scores."
)

ticket_text = st.text_area("✍️ Enter ticket text here:")


# Prediction
if st.button("🔍 Classify Ticket"):

    if not ticket_text.strip():
        st.error("Please enter a ticket message.")
    else:
        top3 = predict_with_explanation(model, ticket_text)

        prediction = top3[0][0]
        confidence = top3[0][1]

        st.subheader("📊 Prediction Result")
        st.write(f"**Category:** {prediction}")
        st.write(f"**Confidence:** {confidence:.2%}")

        # Routing logic
        if confidence >= 0.70:
            st.success("✅ Auto-route to department")
        elif confidence >= 0.40:
            st.warning("⚠️ Send to review queue")
        else:
            st.error("🚨 Escalate to human advisor")

        # Top 3 predictions
        st.subheader("🔎 Top 3 Predictions")

        df = pd.DataFrame(top3, columns=["Category", "Probability"])

        for category, prob in top3:
            st.write(f"{category}: {prob:.2%}")

        st.bar_chart(df.set_index("Category"))

# Footer
st.markdown("---")
st.caption("Built with Scikit-learn + Streamlit")
st.caption("Model: Logistic Regression (selected for probability-based routing)")