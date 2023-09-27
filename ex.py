import streamlit as st

# Create the sidebar menu using with st.columns()
menu_options = ['Moving Averages', 'Data', 'Comparison', 'Predict Future Prices', 'Stock News', 'Description']
menu_selection = st.sidebar.columns(1)[0].selectbox("Stock Dashboard", menu_options)

# Use the with statement to create tabs
with st.container():
    if menu_selection == 'Moving Averages':
        # Insert the main content of the web application
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import plotly.figure_factory as ff
        from pandas_datareader import data as pdr
        import yfinance as yf
        yf.pdr_override()
        from datetime import datetime, timedelta
        from keras.models import load_model

        # Set the start and end dates for the data
        startdate = datetime(2013, 1, 1)
        enddate = datetime(2023, 4, 26)

        st.title('Stock Price Prediction')
        
        # fetch the stock prices using pandas_datareader
        user_input = st.text_input('Enter Stock Ticker')
        if user_input != None:
            try:
                data = pdr.get_data_yahoo(user_input, start=startdate, end=enddate)
                if data.empty:
                    st.warning('No data found for the entered stock ticker. Please try again with a valid ticker.')
                else:
                    st.text(" ")
                    st.subheader('Graphs')
                    st.text(" ")

                    # Visualizations
                    st.subheader('Closing Price vs Time chart')
                    fig = plt.figure(figsize=(12, 6))
                    st.line_chart(data.Close)

                    st.subheader('Closing Price vs Time chart with 100MA')
                    ma100 = data.Close.rolling(100).mean()
                    df = pd.concat([data.Close, ma100], axis=1)
                    df.columns = ['Close', 'ma100']
                    st.line_chart(df)

                    st.subheader('Closing Price vs Time chart with 100MA & 200MA')
                    ma100 = data.Close.rolling(100).mean()
                    ma200 = data.Close.rolling(200).mean()
                    fig = plt.figure(figsize=(12, 6))
                    df1 = pd.concat([data.Close, ma100, ma200], axis=1)
                    plt.plot(ma100)
                    df1.columns = ['Close', 'ma100', 'ma200']
                    st.line_chart(df1)

                    # Split data into training and testing

                    data_training = pd.DataFrame(data['Close'][0:int(len(data) * 0.70)])
                    data_testing = pd.DataFrame(data['Close'][int(len(data) * 0.70): int(len(data))])

                    # Normalize the data using MinMaxScaler
                    from sklearn.preprocessing import MinMaxScaler

                    scaler = MinMaxScaler(feature_range=(0, 1))

                    data_training_array = scaler.fit_transform(data_training)

                    # Load my model
                    model = load_model('keras_model.h5')

                    # Testing Part
                    past_100_days = data_training.tail(100)
                    final_data = past_100_days.append(data_testing, ignore_index=True)
                    input_data = scaler.fit_transform(final_data)

                    x_test = []
                    y_test = []

                    for i in range(100, input_data.shape[0]):
                        x_test.append(input_data[i - 100: i])
                        y_test.append(input_data[i, 0])

                    x_test, y_test = np.array(x_test), np.array(y_test)
                    y_predicted = model.predict(x_test)
                    scaler = scaler.scale_

                    scale_factor = 1 / scaler[0]
                    y_predicted = y_predicted * scale_factor
                    y_test = y_test * scale_factor

                    # Final Graph

                    st.subheader('Predicitons vs Original')
                    # fig2 = plt.figure(figsize=(12,6))
                    combined = np.concatenate([y_test.reshape(-1, 1), y_predicted.reshape(-1, 1)], axis=1)
                    df3 = pd.DataFrame(combined, columns=['Original Price', 'Predicted Price'])
                    st.line_chart(df3)

            except Exception as e:
                st.text("")

    elif menu_selection == 'Data':
        import numpy as np
        import pandas as pd
        from pandas_datareader import data as pdr
        from datetime import datetime
        from keras.models import load_model

        startdate = datetime(2010, 1, 1)
        enddate = datetime(2019, 12, 31)

        st.title('Stock Price Prediction')

        user_input = st.text_input('Enter Stock Ticker')
        if user_input != None:
            try:
                data = pdr.get_data_yahoo(user_input, start=startdate, end=enddate)
                if data.empty:
                    st.warning('No data found for the entered stock ticker. Please try again with a valid ticker.')
                else:
                    # Describing data
                    st.subheader('Data from 2010-2019')
                    st.write(data.describe())

                    data_training = pd.DataFrame(data['Close'][0:int(len(data) * 0.70)])
                    data_testing = pd.DataFrame(data['Close'][int(len(data) * 0.70): int(len(data))])
                    # creating two columns
                    st.info('Same for all DataSets')
                    col1, col2 = st.columns(2)

                    # Display training data
                    with col1:
                        st.subheader('Training Data (70%)')
                        st.write(data_training)

                    # Display testing data
                    with col2:
                        st.subheader('Testing Data (30%)')
                        st.write(data_testing)


            except Exception as e:
                st.text("")

    elif menu_selection == 'Comparison':
        import matplotlib.pyplot as plt
        import yfinance as yf
        from datetime import datetime, timedelta
        from pandas_datareader import data as pdr

        st.title('Stock Price Prediction')

        # set stock symbol
        st.subheader('Comparison between two specific dates')
        user_input = st.text_input('Enter Stock Ticker')

        # Set the minimum and maximum date
        min_date = datetime.now() - timedelta(days=18250)
        max_date = datetime.now() 
        # set start and end dates
        startdate = st.date_input("Enter start date (YYYY-MM-DD): ", min_value=min_date, max_value=max_date)
        enddate = st.date_input("Enter end date (YYYY-MM-DD): ", min_value=min_date, max_value=max_date)


        # Fetching the stock price of date entered by user
        def get_data(user_input, startdate, enddate):
            # Fetch data from Yahoo Finance
            df = yf.download(user_input, start=startdate, end=enddate)  # interval='1d')
            # Extract the closing prices for the selected days
            prices = df['Close']
            # Return the closing prices as a list
            return prices.tolist()


        if user_input and startdate and enddate:
            try:
                prices = get_data(user_input, startdate, enddate)
                st.info(f"The Closing Price of {user_input} on {startdate} was : {prices[0]}")
                st.info(f"The Closing Price of {user_input} on {enddate} is : {prices[-1]}")

                # Display the difference textually
                if prices[0] > prices[-1]:
                    st.info(f'{startdate} has a high stock price when compared to {enddate}')
                elif prices[0] < prices[-1]:
                    st.info(f'{startdate} has a less stock price when compared to {enddate}')
                elif prices[0] == prices[-1]:
                    st.info(f'{startdate} and {enddate} have the equal stock price')

            except:
                st.warning("Invalid Ticker Symbol or Date Range")

        # Fetch stock prices
        if user_input != None:
            try:
                data = pdr.get_data_yahoo(user_input, startdate, enddate)
                if data.empty:
                    st.warning('No data found for the entered stock ticker. Please try again with a valid ticker.')
                else:
                    st.set_option('deprecation.showPyplotGlobalUse', False)

                    # plot closing prices
                    st.line_chart(data.Close)

            except Exception as e:
                st.text("")
                
    elif menu_selection == 'Predict Future Prices':
        import streamlit as st
        import yfinance as yf
        import pandas as pd
        import matplotlib.pyplot as plt
        import plotly.graph_objects as go
        from datetime import datetime, timedelta

        # Function to get moving averages
        def calculate_moving_averages(data):
            data['100-day MA'] = data['Close'].rolling(window=100).mean()
            data['200-day MA'] = data['Close'].rolling(window=200).mean()
            return data

        # Function to predict next two days stock price
        def predict_next_two_days(data):
            last_date = data.index[-1]
            next_day = last_date + timedelta(days=1)
            day_after = last_date + timedelta(days=2)
            prediction = {'Date': [next_day, day_after]}
            
            for col in ['100-day MA', '200-day MA']:
                prediction[col] = data[col].iloc[-1]
            
            return pd.DataFrame(prediction)

        # Streamlit app
        st.title("Future Price Prediction")

        # User input for stock ticker
        ticker = st.text_input("Enter stock ticker")

        if ticker:
            try:
                # Fetch data from Yahoo Finance
                data = yf.download(ticker, start=datetime(2023,1,1), end=datetime.now())

                # Calculate moving averages
                data = calculate_moving_averages(data)

                # Display the data
                st.subheader(f"Stock Data for {ticker}")
                st.write(data)

                # Predict next two days
                prediction = predict_next_two_days(data)
                st.subheader("Predicted Next Two Days")
                st.write(prediction)
                
                # Plotting the next two days
                st.subheader("Next Two Days Prediction Chart")

                fig = go.Figure()

                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
                fig.add_trace(go.Scatter(x=prediction['Date'], y=prediction['100-day MA'], mode='lines', name='100-day MA', line=dict(dash='dash')))
                fig.add_trace(go.Scatter(x=prediction['Date'], y=prediction['200-day MA'], mode='lines', name='200-day MA', line=dict(dash='dash')))

                fig.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Price',
                    title='Next Two Days Prediction',
                    legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
                    autosize=False,
                    width=800,
                    height=500
                )

                st.plotly_chart(fig)

            except Exception as e:
                st.error(f"Error: {e}")


    elif menu_selection == 'Stock News':
        import requests
        from bs4 import BeautifulSoup

        try:
            # Scraping the latest stock news from Yahoo Finance
            url = 'https://finance.yahoo.com/'
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            news_headlines = soup.find_all('h3', {'class': 'Mb(5px)'})

            # Displaying the news headlines using Streamlit
            st.write("## Top Stock News")
            for i, headline in enumerate(news_headlines):
                st.write(f"{i + 1}. {headline.text}")
        except Exception as e:
            st.info("Data cannot be fetched, Check your network connection")
            
    elif menu_selection == 'Description':
        st.header('Moving Averages')
        st.write("The 100-day and 200-day moving averages are common technical indicators used in financial analysis to smooth out price data and identify trends over a specified period of time. Here's how they are calculated:")
        st.subheader("1. 100-Day Moving Average (100-day MA):")
        st.write("  * The 100-day moving average is calculated by taking the average of a stock's closing prices over the last 100 trading days.")
        st.write("  * Mathematically, it is computed as the sum of the closing prices for the past 100 days divided by 100.")
        st.write("Formula:")
        st.info("100-day MA = (∑Closing Pirces for Last 100Days)/100")
        st.write("Purpose:")
        st.write("  * The 100-day moving average is used to smooth out short-term fluctuations in a stock's price. It provides a longer-term view of the stock's trend.")
        
        st.subheader("2. 200-Day Moving Average (200-day MA):")
        st.write("  * The 200-day moving average is calculated by taking the average of a stock's closing prices over the last 200 trading days.")
        st.write("  * Mathematically, it is computed as the sum of the closing prices for the past 200 days divided by 200.")
        st.write("Formula:")
        st.info("200-day MA = (∑Closing Prices for Last 200 Days)/200")
        st.write("Purpose:")
        st.write("*    The 200-day moving average is an even longer-term indicator. It provides a smoothed view of a stock's trend over a more extended period compared to the 100-day moving average.")
        
        st.header('Data')
        st.write('> This data is used to train the LSTM Machine Learning Algorithm. The data is collected from yahoo finance website using yfinance library.')
        st.write("> The data contains 6 attributes those are Open, High, Low, Close, Adj Close, Volume. Here I have used only the Close column to predict the stock prices")
        st.subheader("1. Setting Start and End Dates:")
        st.write("*    startdate and enddate are defined to specify the time range for which the stock data will be retrieved. In this case, it's from January 1, 2010, to December 31, 2019.")
        st.subheader("2. User Input for Stock Ticker:")
        st.write("*    user_input = st.text_input('Enter Stock Ticker') prompts the user to input a stock ticker.")
        st.subheader("3. Fetching Stock Data:")
        st.write("*    data = pdr.get_data_yahoo(user_input, start=startdate, end=enddate) attempts to retrieve historical stock data for the entered stock ticker within the specified date range.")
        st.subheader("4. Describing Data:")
        st.write("*    If data is successfully fetched, it displays summary statistics using st.write(data.describe()).")
        st.subheader("5. Displaying Data:")
        st.write("*    The training and testing data are displayed side by side using st.columns(2) for a two-column layout.")
        st.write("*    In the left column, it shows the training data and in the right column, it shows the testing data.")
        
        st.header("Comparison")
        st.write("In this module we can compare the stock prices of two dates of any company that are listed on yahoo finance website.")
        st.subheader("1. User Input for Stock Ticker and Dates:")
        st.write("*    st.subheader('Comparison between two specific dates') sets a subheader for the user input section.")
        st.write("*    user_input = st.text_input('Enter Stock Ticker') prompts the user to input a stock ticker.")
        st.write('*    startdate = st.date_input("Enter start date (YYYY-MM-DD): ", min_value=min_date, max_value=max_date) and enddate = st.date_input("Enter end date (YYYY-MM-DD): ", min_value=min_date, max_value=max_date) allow the user to input start and end dates for the stock price analysis.')
        st.subheader("2. Fetching Stock Price Data:")
        st.write("*    The get_data() function is defined to fetch stock data for the specified stock ticker and date range.")
        st.write("*    It uses the yf.download() function to retrieve the data and extracts the closing prices.")
        st.subheader("3. Displaying Stock Price Information:")
        st.write("*    It displays the closing price of the specified stock on the entered start and end dates.")
        st.write("*    It also provides textual information about the price difference.")
        st.subheader("4. Fetching and Plotting Stock Prices:")
        st.write("*    If valid user input is provided, it attempts to fetch historical stock data using pdr.get_data_yahoo().")
        st.write("*    If data is successfully retrieved, it plots the closing prices using st.line_chart(data.Close).")
        st.header("Predict Future Prices")
        st.write("> In this module we can predict two days stock prices in advance and plot the interactive graph using st.plotly_chart(fig).")
        st.write("*    It utilizes key libraries like streamlit, yfinance, pandas, matplotlib, and plotly.graph_objects to create an interactive interface. Users begin by inputting a stock ticker. The app fetches historical data from Yahoo Finance, focusing on the period from January 1, 2023, to the present date. It then calculates essential indicators: the 100-day moving average (100-day MA) and the 200-day moving average (200-day MA). These smoothed averages provide valuable insights into the stock's trend over their respective time frames.")
        st.write("*    The app also displays the acquired stock data, along with the calculated moving averages. This visual representation gives users a clear overview of the stock's historical performance. Additionally, it predicts the stock prices for the next two days based on the latest data. These projections are presented in a concise tabular format, with dates aligned to the corresponding prices. The app enhances user experience through interactive plotting with Plotly. It creates a dynamic line chart showcasing the closing prices, 100-day MA, and 200-day MA. This graphical representation offers a more intuitive understanding of the stock's behavior. In case of any unexpected errors, the app handles them gracefully by providing an informative error message. Overall, this Streamlit app provides a user-friendly platform for exploring and predicting future stock prices, offering valuable insights for investors and traders.")
        st.header("Stock News")
        st.write("> It displays the top 10 stock news from yahoo finance. The news is webscrapped from the yahoo finance website using BeautifulSoap library.")
        st.subheader("1. Web Scraping:")
        st.write("*    The code starts by attempting to scrape the latest stock news from Yahoo Finance. It sends a GET request to the Yahoo Finance URL and stores the response.")
        st.subheader("2. Parsing HTML Content:")
        st.write("*    The HTML content of the response is then parsed using BeautifulSoup. This allows the code to easily navigate and extract specific elements from the webpage.")
        st.subheader("3. Finding News Headlines:")
        st.write("*    The code uses BeautifulSoup to find all the <h3> elements with the class 'Mb(5px)'. These elements typically contain the headlines of the latest news articles on Yahoo Finance.")
        st.subheader("4. Displaying News Headlines:")
        st.write('*    After extracting the news headlines, the code uses Streamlit to display them. It first prints the heading "Top Stock News" using st.write("## Top Stock News"). Then, it iterates through the list of headlines and displays them with their corresponding index.')        
