import streamlit as st
import pickle
import numpy as np

with open("classifier.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("classifier_all.pkl", "rb") as model_file:
    model_all = pickle.load(model_file)

st.image("trace_logo.jpg", width=200)
st.title("Overall rating Classifier")
st.write("Enter the ratings for the specific aspect to predict the overall rating.")

tab1, tab2 = st.tabs(["3 inputs", "7 inputs"])

with tab1:
    st.header("Inputs")
    
    value_for_money = st.slider("Value for Money", key = 2, min_value=0, max_value=5, step=1)
    cabin_staff_service = st.slider("Cabin Staff - Service", key = 3, min_value=0, max_value=5, step=1)
    seat_comfort = st.slider("Seat and Comfort", key = 4, min_value=0, max_value=5, step=1)


    if st.button("Predict", key = 1):
        features = np.array([[value_for_money, cabin_staff_service, seat_comfort]])
        prediction = model.predict(features)
        positive = 'Positive overall satisfaction'
        negative = 'Negative overall satisfaction'
        text_prediction = np.where(prediction == 1, positive, negative)
        st.write(f"Predicted: {text_prediction}")
        # st.write(f'where 1 is positive / 0 is negative for overall satisfaction')



with tab2:
    st.header("Inputs")
    
    col1, col2 = st.columns(2, border=True)
    with col1:
        value_for_money = st.slider("Value for Money", key = 72, min_value=0, max_value=5, step=1)
        ground_service = st.slider("Ground Staff - Service", key = 73, min_value=0, max_value=5, step=1)
        cabin_staff_service = st.slider("Cabin Staff - Service", key = 74, min_value=0, max_value=5, step=1)
    with col2:
        seat_comfort = st.slider("Seat and Comfort", key = 75, min_value=0, max_value=5, step=1)
        food_and_beverages = st.slider("Food and Beverages", key = 76, min_value=0, max_value=5, step=1)
        inflight_entertainment = st.slider("Inflight Entertainment", key = 77, min_value=0, max_value=5, step=1)
        wifi_rating = st.slider("WiFi and Connectivity", key = 78, min_value=0, max_value=5, step=1)

    if st.button("Predict", key = 71):
        features = np.array([[value_for_money, ground_service, cabin_staff_service, seat_comfort, 
                              food_and_beverages, inflight_entertainment, wifi_rating]])
        prediction = model_all.predict(features)
        positive = 'Positive overall satisfaction'
        negative = 'Negative overall satisfaction'
        text_prediction = np.where(prediction == 1, positive, negative)
        st.write(f"Predicted: {text_prediction}")
        # st.write(f'where 1 is positive / 0 is negative for overall satisfaction')
        with st.popover("Info"):
            st.markdown("Feature importance of 7 inputs")
            st.image("Feature-importance.jpg", width=1200)



# [theme]
# primaryColor = "#041978"
# backgroundColor = "#FFFFFF"
# secondaryBackgroundColor = "#F0F0F0"
# textColor = "#000000"
# [theme]
# primaryColor = "#F39C12"
# backgroundColor = "#FFFFFF"
# font = "sans serif"

# st.markdown(
    
#     <style>
#     .reportview-container {
#         background: url("trace_logo.jpg");
#     }
#    </style>
#     ,
#     unsafe_allow_html=True
# )
