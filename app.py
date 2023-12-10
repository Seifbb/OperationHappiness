import streamlit as st
import seaborn as sns
from prophet import Prophet
import pickle
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Load your trained model
with open('../prophet_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load data
df = pd.read_csv("../happiness_data.csv")

# App title
st.title('Happiness Score Forecasting App')

# User input for the date
user_input_date = st.date_input("Enter a Date for Prediction", min_value=datetime.now())


def predict_happiness_score(date):
    
    last_data_date = pd.to_datetime(df.at[df.shape[0]-1,"date"]) 
    future = model.make_future_dataframe(periods= (pd.to_datetime(date)-last_data_date).days ) 
    forecast = model.predict(future)    

    return forecast#['yhat'].iloc[0]

if 'prediction_made' not in st.session_state:
    st.session_state['prediction_made'] = False
    
if 'figure_generated' not in st.session_state:
    st.session_state['figure_generated'] = False

if st.button('Predict'):
    # Perform prediction
    forecast = predict_happiness_score(user_input_date)
    prediction = forecast['yhat'].iloc[-1]
    # Display the prediction
    st.write(f'Predicted Happiness Score on {user_input_date}: {prediction}')
        
    # Plotting
    fig = model.plot(forecast)
    ax = fig.gca()
    ax.axvline(user_input_date, color='r', linestyle='--')
    ax.text(user_input_date, forecast['yhat'].max(), 'Prediction', color='red')
    st.pyplot(fig)
    
    # Store the prediction in session state
    st.session_state['prediction'] = prediction
    st.session_state['forecast'] = forecast
    
    # Update the session state
    st.session_state['prediction_made'] = True
    
    # This is for keeping the first figure throughout code execution
    st.session_state['figure_generated'] = True
    

    
    
if st.session_state['prediction_made']:
    if st.button('Go Further?'):
        
        if st.session_state.get('figure_generated', False):
    
            forecast = st.session_state['forecast']
            prediction = st.session_state['prediction']
        
            st.write(f'Predicted Happiness Score on {user_input_date}: {prediction}')
    
            # Plotting
            fig = model.plot(forecast)
            ax = fig.gca()
            ax.axvline(user_input_date, color='r', linestyle='--')
            ax.text(user_input_date, forecast['yhat'].max(), 'Prediction', color='red')
            st.pyplot(fig)
        
        prediction = st.session_state['prediction']

        fig2, ax2 = plt.subplots()
        sns.histplot(df['score'], ax=ax2, kde=True)

        # Highlighting the forecasted value
        ax2.axvline(prediction, color='r', linestyle='--')
        ax2.text(prediction, ax2.get_ylim()[1], 'Forecasted Value', color='red')
        st.pyplot(fig2)
        
        
        # Additional insights
        mean_value = df['score'].mean()
        quantile = df['score'].quantile([0.25, 0.5, 0.75])
        quantile_info = "below the 25th percentile. There is clearly a room for salary raise here."
        if prediction > quantile[0.75]:
            quantile_info = "above the 75th percentile. Life is good"
        elif prediction > quantile[0.5]:
            quantile_info = "between the 50th and 75th percentiles. Organizing a night out in bars is still cool though"
        elif prediction > quantile[0.25]:
            quantile_info = "between the 25th and 50th percentiles. Start considering a bowling/pizza night"

        st.write(f'The forecasted value is {"higher" if prediction > mean_value else "lower"} than the average.')
        st.write(f'The forecasted value is {quantile_info}.')
    
    
# Optional: Add some explanations or descriptions
st.markdown("""
This app predicts the happiness score based on historical data using a machine learning model.
""")
