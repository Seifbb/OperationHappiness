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

# Prediction function for convenience
def predict_happiness_score(date):
    
    last_data_date = pd.to_datetime(df.at[df.shape[0]-1,"date"]) 
    future = model.make_future_dataframe(periods= (pd.to_datetime(date)-last_data_date).days ) 
    forecast = model.predict(future)    

    return forecast#['yhat'].iloc[0]


# Handling session states for button correctness
if 'prediction_made' not in st.session_state:
    st.session_state['prediction_made'] = False
    
if 'figure_generated' not in st.session_state:
    st.session_state['figure_generated'] = False
    
if 'go_further' not in st.session_state:
    st.session_state['go_further'] = False
    
if 'explore_dataset' not in st.session_state:
    st.session_state['explore_dataset'] = False
    
if 'explore_option' not in st.session_state:
    st.session_state['explore_option'] = None
    
if 'show_date_filter' not in st.session_state:
    st.session_state['show_date_filter'] = False
    
    
############################################################################################    
# First button: prediction upon date selection
############################################################################################   

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
    
############################################################################################    
# Second button: statistical analysis
############################################################################################
    
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
        
        # Updating session state
        st.session_state['go_further'] = True

                
############################################################################################    
# Third button: filter by dates and by score
############################################################################################      
        
        
if st.session_state['go_further']:
        
    if st.button('Explore Dataset'):
        st.session_state['explore_dataset'] = True
        
    if (st.session_state['explore_dataset']):
        st.session_state['explore_option'] = st.selectbox(
            'Choose an option to explore the dataset',
            ['Filter by Score', 'Filter by Dates'], 
            index=0 if st.session_state['explore_option'] is None else 1
        )
        
        if st.session_state['explore_option'] == 'Filter by Score':
            # Score selection
            selected_score = st.number_input('Enter a Score', value=st.session_state['prediction']) #min_value=0.0, max_value=100.0, step=0.1)
            if selected_score:
                
                    forecast = st.session_state['forecast']
                    tolerance = 0.01
                    score_min = selected_score - tolerance
                    score_max = selected_score + tolerance

                    # Filter the DataFrame
                    filtered_df = df[(df['score'] >= score_min) & (df['score'] <= score_max)]

                    # Get the dates
                    dates = filtered_df['date']
                    st.write(f'Similar scores were obtained {len(dates.values)} times in: {dates.values}')

                    # Plotting
                    fig = model.plot(forecast)
                    ax = fig.gca()
                    ax.axhline(selected_score, color='g', linestyle='--')
                    ax.text(user_input_date, forecast['yhat'].max(), 'Prediction', color='red')
                    st.pyplot(fig)


        elif st.session_state['explore_option'] == 'Filter by Dates':

            # User selects a date range
            start_date = st.date_input('Start Date', value=pd.to_datetime(df['date'].min()))
            end_date = st.date_input('End Date', value=pd.to_datetime(df['date'].max()))
            
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            if start_date and end_date:
                st.write("Dates chosen.")
                
                
                df["date"] = pd.to_datetime(df["date"] )
                filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                fig3, ax3 = plt.subplots()
                plt.plot(filtered_df['date'], filtered_df['score'])
                plt.xlabel('Date')
                plt.ylabel('Score')
                plt.title('Time Series Plot of Happiness Score')
                plt.grid(True)
                plt.show()
                
                fig, ax = plt.subplots()
                ax.plot(filtered_df['date'], filtered_df['score'])
                ax.set_xlabel('Date')
                ax.set_ylabel('Score')
                ax.set_title('Time Series Plot of Happiness Score')
                ax.grid(True)

                st.pyplot(fig)
                
        


    
    
# Optional: Add some explanations or descriptions
st.markdown("""
This app predicts the happiness score based on historical data using a machine learning model.
""")
