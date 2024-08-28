import streamlit as st
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def main():
    perform_calcs= False #variable to determine if it's time to run the model
    st.title(':blue[Stock Price Predictor]')
    ticker = st.text_input("Enter Stock Ticker Symbol", placeholder="E.g. AAPL").upper()
    if len(ticker) > 0:
        #obtain key dates to use with yahoo finances
        today=datetime.today()
        end_date = str(today)
        end_date= end_date[:10]
        one_month_ago = today - relativedelta(months=1)
        start_date = str(one_month_ago)
        start_date = start_date[:10]
        
        info = yf.download(f'{ticker}', start=f'{start_date}', end=f'{end_date}')
        if info.empty:
            st.error('Invalid ticker symbol')
        else: 
            st.divider()
            st.write(f"Here's some **{ticker}** data for the past month")
            info
            fig= plt.figure(figsize=(16,8))
            plt.title('Closing Price History')
            plt.plot(info['Close'])
            plt.xlabel('Date', fontsize= 18)
            plt.ylabel('Close Price', fontsize= 18)
            st.pyplot(fig=fig)
            st.divider()
            st.write("**Now let's see how our model predicts prior closing prices**")
            with st.expander("Choose Start Date for Prediction"):
                st.write("The model trains on closing prices that are 60 days prior to the target day. Therefore, the minimum start date must be at least 60 days ago, with more days leading to more accurate predictions. Please check to make sure your start day is valid for your stock. Defaulted to three years ago.")
                sixty_days_ago = today - timedelta(days=60)
                three_years_ago= today - timedelta(days=1095)
                start= st.date_input("Start Date", value=three_years_ago, max_value=sixty_days_ago)
                df = yf.download(f'{ticker}', start=f'{start}', end=f'{end_date}')
                if df.empty:
                    st.error("Invalid start date entered")
                else:
                    if st.button("Submit"):
                        perform_calcs= True
            if perform_calcs== True:
                with st.spinner("Wait for it..."):
                    #Create a new df with only Closing data
                    data=df.filter(['Close'])
                    #convert df to a numpy array
                    dataset= data.values
                    #get the numbner of rows to train the model on (80% of data will be used for training)
                    training_data_length= math.ceil(len(dataset)*.8)
                    #scale the data from 0-1 before inputting to neural network
                    scaler = MinMaxScaler(feature_range=(0,1))
                    scaled_data = scaler.fit_transform(dataset)
                    
                    #create a training dataset

                    #create a scaled training dataset with 80% of original data
                    training_data = scaled_data[0:training_data_length, :]

                    #split the data into x and y training datasets
                    x_train=[] #independent training features
                    y_train=[] #dependent training features

                    for i in range(60, len(training_data)):
                        x_train.append(training_data[i-60:i, 0]) #Contains sequences of 60 past data points (features) for each sample, not including i.
                        y_train.append(training_data[i, 0]) #Contains the next data point (target) following each sequence in x_train.
                    
                    #convert training datasets to numpy arrays
                    x_train, y_train = np.array(x_train), np.array(y_train)
                    #reshape the data because LSTM model needs 3 dimensional data (#of samples, #of time stamps, and #of features)
                    x_train=np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                    #Build the LSTM model
                    model = Sequential()
                    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                    model.add(LSTM(50, return_sequences=False))
                    model.add(Dense(25))
                    model.add(Dense(1))
                    #compile the model
                    model.compile(optimizer='adam', loss= 'mean_squared_error') #optimizer used to improve (minimize) loss function, loss function used to measure how the model did on training
                    #training the model
                    model.fit(x_train, y_train, batch_size= 1, epochs= 1)

                    #create the testing dataset

                    #create a new array containing scaled values using the rest of the original data
                    test_data= scaled_data[training_data_length - 60:, :]

                    x_test = []
                    y_test = dataset[training_data_length:, :] #values we want the model to predict

                    for i in range(60, len(test_data)):
                        x_test.append(test_data[i-60:i, 0])

                    #convert data to numpy array
                    x_test = np.array(x_test)
                    #reshape the test data
                    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
                    #Get the model's predictions
                    predictions= model.predict(x_test)
                    predictions = scaler.inverse_transform(predictions)
                    #evaluate model through RMSE
                    rmse = np.sqrt(np.mean(predictions-y_test)**2)
                    #plotting the data
                    train= data[:training_data_length]
                    valid= data[training_data_length:]
                    valid['Predictions'] = predictions

                    fig2 = plt.figure(figsize=(16,8))
                    plt.title(f'Model with RMSE {rmse}')
                    plt.xlabel('Date', fontsize= 18)
                    plt.ylabel('Closing Price (USD)', fontsize= 18)
                    plt.plot(train['Close'])
                    plt.plot(valid[['Close', 'Predictions']])
                    plt.legend(['Training Data', 'Actual Values', 'Predictions'], loc="lower right")
                    st.pyplot(fig=fig2)
                    #show the actual and predicted prices
                    st.divider()
                    st.write("Here are the actual and predicted closing prices on a table.")
                    valid
                    #try to predict a day forward
                    #get the quote
                    stock_quote = yf.download(f'{ticker}', start=f'{start}', end=f'{end_date}')
                    new_df= stock_quote.filter(['Close'])

                    #get the last 60 day closing price values and store as array
                    last_60_days = new_df[-60:].values
                    #scaling the data
                    last_60_days_scaled= scaler.transform(last_60_days)

                    X_test= []
                    X_test.append(last_60_days_scaled)
                    X_test= np.array(X_test)
                    X_test= np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                    predicted_price = model.predict(X_test)
                    predicted_price = scaler.inverse_transform(predicted_price)
                    st.write(f"**The predicted stock price for a day forward is {predicted_price} USD**")

    

    
   
    

    
    

if __name__ == "__main__":
    main()