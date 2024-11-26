import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# from data_fetcher import fetch_stock_data, get_real_time_data, fetch_data
import plotly.graph_objects as go
from helper import * 
from prophet import Prophet
from prophet.plot import plot_plotly
from statsmodels.tsa.arima.model import ARIMA



end_date = pd.to_datetime('today').strftime('%Y-%m-%d')
company_names = {
    'Tesla, Inc.': 'TSLA',
    'Apple Inc.': 'AAPL',
    'Amazon.com, Inc.': 'AMZN',
    'Alphabet Inc. (Google)': 'GOOG',
    'Microsoft Corporation': 'MSFT',
    'Meta Platforms, Inc. (Facebook)': 'META',
    'NVIDIA Corporation': 'NVDA',
    'Berkshire Hathaway Inc.': 'BRK.A',
    'Netflix, Inc.': 'NFLX',
    'PayPal Holdings, Inc.': 'PYPL'
}


def main():

    st.set_page_config(
        page_title="Stock Price Prediction",
        page_icon="📈"
    )

    # STREAMLIT UI
    st.title('Stock Price Prediction')
    st.sidebar.header("User Input Features")

    st.sidebar.subheader("Select stock")
    selected_company = st.sidebar.selectbox("Choose a stock", list(company_names.keys()))
    
    selected_ticker = company_names[selected_company]
    st.sidebar.subheader("Stock Ticker")
    ticker = st.sidebar.text_input("Stock ticker code ", selected_ticker, disabled=True)

    periods = get_periods_intervals()
    
    st.sidebar.subheader("Select period")
    selected_period = st.sidebar.selectbox("Choose a period", list(periods.keys()))

    st.sidebar.subheader("Select interval")
    selected_interval = st.sidebar.selectbox("Choose an interval", periods[selected_period])

    if ticker:

        fig = go.Figure()
        # df = get_real_time_data(ticker=ticker, period=selected_period, interval=selected_interval)
        df = fetch_stock_data(ticker, period=selected_period, interval=selected_interval)

        st.subheader("Historical Data")

        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close']))
        fig.update_layout(
        title=f"Historical {ticker} Stock Data",
        xaxis_rangeslider_visible=False,
        xaxis_title='Date',
        yaxis_title='Closing Price (USD)',
        template="plotly_dark" , 
        
    )
        st.plotly_chart(fig)

        # Let the user choose which model they want to visualize the prediction

        model_choice = st.radio("Select the model to visualize:", ('Prophet Model', 'AutoRegressive (AR) Model'))

        st.subheader(f"Stock Prediction uding {model_choice}")

        if model_choice == 'Prophet Model':
        
        # Unpack data
            data = fetch_stock_data(ticker=ticker, period="2y", interval="1d")
            
            # make a copy of the data(not really necessary)
            df = data.copy()

            #  Reset the index so prophet can recognize the column
            df=df.reset_index()

            df=df.rename(columns={'Date': 'ds', 'Close': 'y'})

            # Remove timezone if its present
            df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
            
            # Initialize prophet model
            m = Prophet()

            # Fit the model to historical data
            m.fit(df)

            # Create dataframe for future predictions
            future = m.make_future_dataframe(periods=365)

            # Make predictions 
            forecast = m.predict(future)

            fig1 = plot_plotly(m, forecast)
        
            fig1.update_layout(
                xaxis_title="Date",
                yaxis_title="Stock Price (USD)",
            )

            # Customize the color of the forecasted predictions (trend, uncertainty intervals, etc.)
            
            st.plotly_chart(fig1)
        else:
            
            # Unpack data
            train_data, test_data, forecast, predictions = get_stock_predictions(ticker)

            # Check if the data is not None
            if train_data  is not None and (forecast >= 0).all() and (predictions >= 0).all():

                fig = go.Figure(
                    data=[
                        go.Scatter(
                            x=train_data.index,
                            y=train_data['Close'],
                            name='Train',
                            mode='lines',
                            line=dict(color='blue')
                        ),
                        go.Scatter(
                            x=test_data.index,
                            y=test_data['Close'],
                            name='Test',
                            mode='lines',
                            line=dict(color='orange')
                        ),
                        go.Scatter(
                            x=forecast.index,
                            y=forecast,
                            name='Forecast',
                            mode='lines',
                            line=dict(color='red')
                        ),
                        go.Scatter(
                            x=test_data.index,
                            y=predictions,
                            name='Test Predictions',
                            mode='lines',
                            line=dict(color='green')
                        )
                    ]
                )
                fig.update_layout(xaxis_rangeslider_visible=False)

        # Use the native streamlit theme.
                st.plotly_chart(fig, use_container_width=True)

            # If the data is None
            else:
                # Add a title to the stock prediction graph
                st.markdown("## **Stock Prediction**")

                # Add a message to the stock prediction graph
                st.markdown("### **No data available for the selected stock**")

        

        # model = ARIMA(df['y'], order=(16,0,1))
        # model_fit = model.fit()

        # forecast_steps = 180  # forecasting for the next 365 days
        # forecast = model_fit.forecast(steps=forecast_steps)
        # forecast_index = pd.date_range(start=df['ds'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
        
        # forecast_df = pd.DataFrame({
        #     'ds': forecast_index,
        #     'yhat': forecast
        # })
        # forecast_fig = go.Figure()
        # forecast_fig.add_trace(go.Scatter(
        #     x=df['ds'],
        #     y=df['y'],
        #     mode='lines',
        #     name='Historical prices',
        #     line=dict(color='blue')

        # ))
        # forecast_fig.add_trace(go.Scatter(
        #     x=forecast_df['ds'],
        #     y=forecast_df['yhat'],
        #     mode='lines',
        #     name="Forecast",
        #     line=dict(color='red')
        # ))
        
        # forecast_fig.update_layout(
        #     title=f'{ticker} Stock Price Prediction',
        #     xaxis_title='Date',
        #     yaxis_title='Price (USD)',
        #     template='plotly_dark',
        # )

        # st.plotly_chart(forecast_fig)
    # if ticker:
    #     st.write(f"Fetching real-time-data for {ticker}")
    #     df = get_real_time_data(ticker)
    #     if df.empty:
    #         st.write("No data available. Please try again later.")
    #     else:
    #         st.subheader(f"Real-Time Data for {ticker}")
    #         st.write(df.tail(10))

    #         # Train ARIMA model on recent data
    #         st.subheader("Training ARIMA Model on Latest Data..")
    #         model = train_arima_model(df)

    #         # Forecast the next minutes closing price
    #         forecasted_price = forecast_arima(model, steps=1).iloc[0]
    #         print(forecasted_price)
    #         st.subheader(f"Predicted next minute closing price: ${forecasted_price:.2f}")

    #         # Plot the most recent stock data

    #         st.subheader("Real-Time Stock Price Trend")
    #         plt.figure(figsize=(10, 6))
    #         plt.plot(df.index, df["Close"], label="Real-Time Closing Price")
    #         plt.title(f"Real-Time Stock Price Trend for {ticker}")
    #         plt.xlabel("Time")
    #         plt.ylabel("Price (USD)")
    #         plt.legend()
    #         st.pyplot(plt)

if __name__ == "__main__":
    main()