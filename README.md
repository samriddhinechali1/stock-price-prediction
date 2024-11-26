# 📊 Stock Price Prediction App 📈

This app predicts the future stock prices of various companies using advanced machine learning models. With this tool, you can analyze historical data and make predictions using two popular models: **Prophet** and **AutoRegressive (AR)**.Currently, the app supports predictions for only 10 companies but there might be changes in the future! 

## 🧰 Models used

The following frameworks and modules have been used to make this app-

- **Streamlit**: For building the interactive web interface and displaying data 
- **Pandas**: For data manipulation and handling stock data 
- **Plotly**: For creating interactive and beautiful data visualizations 
- **Prophet**: For time series forecasting and stock price prediction 
- **Statsmodels**: For statistical modeling, including AutoRegressive (AR) and other time series models 
- **yfinance**: For fetching real-time stock data 


## 🔑 Key Features
* **Real-Time Data**: Fetch live stock data for selected companies 
* **Visualizing Forecast**: Display future stock price predictions alongside historical data 
* **Prophet Model**: Make time-series predictions with the powerful Prophet model 
* **AutoRegressive (AR) Model**: Predict stock trends using the statistical AR model from Statsmodels 
* **Customizable Settings**: Select the stock, time period, and data interval for flexible analysis


## 🌟 How to Run the App Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/samriddhinechali1/stock-price-prediction.git
   ```
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run stock_predictor.py
   ```


## 🚀 Conclusion
This app provides a simple yet powerful tool for predicting stock prices using time-series forecasting techniques. While I’ve used the Prophet model and the AutoRegressive (AR) model to generate predictions, I have also experimented with the ARIMA model. However, the results from ARIMA were unsatisfactory, so I decided not to include it in the app for now. Despite this, I plan to revisit ARIMA or explore other models like LSTM (Long Short-Term Memory) networks to further improve accuracy in future updates.

## 📌 Note
Remember, this app is just for fun—don’t bet your life savings on these predictions! Real stock market decisions should always come with solid research and professional advice. 😉


