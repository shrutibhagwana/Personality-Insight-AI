import streamlit as st
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
from PIL import Image

# ------------------------------
# Page Settings
# ------------------------------
st.set_page_config(
    page_title="Personality Insight AI",
    page_icon="🧠",
    layout="wide"
)

# ------------------------------
# Load CSS
# ------------------------------
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except:
    pass

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_files():
    model = tf.keras.models.load_model("personality_ann_model.keras")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, scaler, encoder

model, scaler, encoder = load_files()

# ------------------------------
# Sidebar
# ------------------------------

st.sidebar.title("🧠 Personality AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔮 Predict",
        "📊 Performance",
        "ℹ️ About",
        "👩 Developer"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Deep Learning Project")
st.sidebar.caption("ANN Based Personality Prediction")

# ------------------------------
# HOME PAGE
# ------------------------------

if page == "🏠 Home":

    st.title("🧠 Personality Insight AI")

    st.markdown("---")

    col1, col2 = st.columns([2,1])

    with col1:

        st.header("Welcome 👋")

        st.write("""
This application predicts a person's personality using
Artificial Neural Networks (ANN).

The model classifies personality into:

✅ Introvert

✅ Extrovert

✅ Ambivert
""")

        st.info("Click on **Predict** from the sidebar to test the model.")

    with col2:

        st.markdown(
        """
        # 🧠

        ## AI Personality

        ### Prediction System
        """
        )

    st.markdown("---")

    c1,c2,c3=st.columns(3)

    with c1:
        st.metric("Accuracy","99.70%")

    with c2:
        st.metric("Classes","3")

    with c3:
        st.metric("Algorithm","ANN")

# ------------------------------
# PERFORMANCE PAGE
# ------------------------------

elif page=="📊 Performance":

    st.title("📊 Model Performance")

    st.markdown("### Overall Performance")

    st.metric("Accuracy","99.70 %")
    st.metric("Loss","0.0174")

    st.markdown("---")

    st.write("### Model Details")

    st.write("""
**Algorithm :** Artificial Neural Network

**Hidden Layers :** 2

**Optimizer :** Adam

**Activation Function :** ReLU

**Output Layer :** Softmax

**Epochs :** 30

**Batch Size :** 32
""")

    st.success("The model achieved high accuracy on the test dataset.")
# ------------------------------
# ABOUT PAGE
# ------------------------------

elif page=="ℹ️ About":

    st.title("ℹ️ About Project")

    st.write("""
### Personality Insight AI

This project predicts a person's personality
using Deep Learning.

The model is trained using
Artificial Neural Networks (ANN).

### Technologies Used

✔ Python

✔ TensorFlow

✔ Keras

✔ Streamlit

✔ NumPy

✔ Pandas

✔ Scikit-Learn

### Output Classes

😊 Extrovert

📚 Introvert

😎 Ambivert
""")
    
# ------------------------------
# DEVELOPER PAGE
# ------------------------------

elif page=="👩 Developer":

    st.title("👩 About Developer")

    st.write("### Shruti")

    st.write("BCA Student of IITM, GGSIPU")

    st.write("This Project Is on Artificial Intelligence & Machine Learning")

    st.markdown("---")

    st.write("### Skills")

    st.write("✔ Python")

    st.write("✔ Machine Learning")

    st.write("✔ Deep Learning")

    st.write("✔ Web Development")

    st.success("Made with ❤️ using Streamlit")

# ------------------------------
# PREDICTION PAGE
# ------------------------------

elif page == "🔮 Predict":

    st.title("🔮 Personality Prediction")

    st.write("Enter the values between **1 to 10**")

    col1, col2 = st.columns(2)

    with col1:

        social_energy = st.slider("Social Energy",1,10,5)
        talkativeness = st.slider("Talkativeness",1,10,5)
        group_comfort = st.slider("Group Comfort",1,10,5)
        listening_skill = st.slider("Listening Skill",1,10,5)
        empathy = st.slider("Empathy",1,10,5)
        creativity = st.slider("Creativity",1,10,5)
        organization = st.slider("Organization",1,10,5)
        leadership = st.slider("Leadership",1,10,5)
        risk_taking = st.slider("Risk Taking",1,10,5)
        curiosity = st.slider("Curiosity",1,10,5)
        excitement_seeking = st.slider("Excitement Seeking",1,10,5)
        friendliness = st.slider("Friendliness",1,10,5)
        planning = st.slider("Planning",1,10,5)
        reading_habit = st.slider("Reading Habit",1,10,5)
        online_social_usage = st.slider("Online Social Usage",1,10,5)

    with col2:

        alone_time_preference = st.slider("Alone Time Preference",1,10,5)
        deep_reflection = st.slider("Deep Reflection",1,10,5)
        party_liking = st.slider("Party Liking",1,10,5)
        public_speaking_comfort = st.slider("Public Speaking Comfort",1,10,5)
        routine_preference = st.slider("Routine Preference",1,10,5)
        emotional_stability = st.slider("Emotional Stability",1,10,5)
        spontaneity = st.slider("Spontaneity",1,10,5)
        adventurousness = st.slider("Adventurousness",1,10,5)
        sports_interest = st.slider("Sports Interest",1,10,5)
        travel_desire = st.slider("Travel Desire",1,10,5)
        gadget_usage = st.slider("Gadget Usage",1,10,5)
        work_style_collaborative = st.slider("Collaborative Work Style",1,10,5)
        decision_speed = st.slider("Decision Speed",1,10,5)
        stress_handling = st.slider("Stress Handling",1,10,5)

    if st.button("🔍 Predict Personality"):

        user_data = [[

            social_energy,
            alone_time_preference,
            talkativeness,
            deep_reflection,
            group_comfort,
            party_liking,
            listening_skill,
            empathy,
            creativity,
            organization,
            leadership,
            risk_taking,
            public_speaking_comfort,
            curiosity,
            routine_preference,
            excitement_seeking,
            friendliness,
            emotional_stability,
            planning,
            spontaneity,
            adventurousness,
            reading_habit,
            sports_interest,
            online_social_usage,
            travel_desire,
            gadget_usage,
            work_style_collaborative,
            decision_speed,
            stress_handling

        ]]

        # Scale input
        user_data = scaler.transform(user_data)

        # Predict
        prediction = model.predict(user_data)

        predicted_index = np.argmax(prediction)

        personality = encoder.inverse_transform([predicted_index])[0]

        confidence = np.max(prediction) * 100

        st.success("Prediction Completed ✅")

        st.markdown("---")

        st.subheader("🎯 Prediction Result")

        if personality == "Extrovert":

            st.success("😊 Personality : Extrovert")

            st.write("This person enjoys social interaction and likes spending time with people.")

        elif personality == "Introvert":

            st.info("📚 Personality : Introvert")

            st.write("This person prefers calm environments and enjoys spending time alone.")

        else:

            st.warning("😎 Personality : Ambivert")

            st.write("This person has a balanced personality and can adapt to different situations.")

        st.write(f"### Confidence : {confidence:.2f}%")