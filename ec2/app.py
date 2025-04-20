import streamlit as st
import requests

st.set_page_config(page_title="Iris Classifier 🌸", layout="centered")

# Sidebar for server input
st.sidebar.title("🔧 API Settings")
server_url = st.sidebar.text_input(
    "FastAPI Server URL (e.g. http://<public-ip>:8000)",
    value="http://localhost:8000"
)

# Main UI
st.title("🌸 Iris Flower Classifier")
st.markdown("Predict the Iris flower species by entering the measurements below.")

# Spacing
st.markdown("---")

# Two-column layout for inputs
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.4)
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.3)

with col2:
    sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.4)
    petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2)

# Predict Button
if st.button("🔍 Predict"):
    features = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }

    try:
        response = requests.post(f"{server_url}/predict", json=features)
        response.raise_for_status()
        result = response.json()
        st.success(f"🌼 Predicted species: **{result['prediction'].title()}**")
    except Exception as e:
        st.error(f"❌ API call failed: {e}")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and FastAPI")
