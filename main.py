import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet
import requests
import sys
from fbprophet.plot import add_changepoints_to_plot

def getDailyData(nemo, outputsize):
  url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + \
      nemo + '&outputsize=' + outputsize + '&apikey=FR61ESIVDJD2UGI8'

  print(url)

  r = requests.get(url)
  dataDaily = r.json()

  print(dataDaily)

  dailyDataDictionary  = dataDaily['Time Series (Daily)']

  df = pd.DataFrame.from_dict(dailyDataDictionary, orient='index')

  df = df.reset_index()
  df = df.rename(index=str, columns={"index": "Date", "1. open": "Open",
                                        "2. high": "High", "3. low": "Low", "4. close": "Close"})
  df['Date'] = pd.to_datetime(df['Date'])

  df = df.sort_values(by=['Date'])

  df.Open = df.Open.astype(float)
  df.Close = df.Close.astype(float)
  df.High = df.High.astype(float)
  df.Low = df.Low.astype(float)

  return df


def getIntradayData(nemo, interval):
  #TODO refactor extract method
  url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=' + \
      nemo + '&interval=' + interval + '&apikey=FR61ESIVDJD2UGI8'
  print(url)
  r = requests.get(url)
  dataIntraday = r.json()
  dataDict = dataIntraday['Time Series (5min)']

  df = pd.DataFrame.from_dict(dataDict, orient='index')
  df = df.reset_index()
  df = df.rename(index=str, columns={"index": "Date", "1. open": "Open",
                                        "2. high": "High", "3. low": "Low", "4. close": "Close"})
  df['Date'] = pd.to_datetime(df['Date'])

  df = df.sort_values(by=['Date'])

  df.Open = df.Open.astype(float)
  df.Close = df.Close.astype(float)
  df.High = df.High.astype(float)
  df.Low = df.Low.astype(float)

  return df

def getDailyDataFromMarketstack():
  params = {
    'access_key': '6149e9af91483588ecfb15ce01dd434d',
    'symbols': 'FALABELLA.XSGO',
    'sort': 'DESC',
    'date_from': '2017-07-13',
    'limit': '1000'
  }

  api_result = requests.get('http://api.marketstack.com/v1/eod', params)

  api_response = api_result.json()

  dataDict = api_response['data']

  df = pd.DataFrame.from_dict(dataDict)
  df = df.reset_index()

  df = df.rename(index=str, columns={"date": "Date", "open": "Open",
                                        "high": "High", "low": "Low", "close": "Close"})
                                  
  df['Date'] = pd.to_datetime(df['Date'])
  df['Date'] = df['Date'].dt.tz_localize(None)

  df = df.sort_values(by=['Date'])

  df.Open = df.Open.astype(float)
  df.Close = df.Close.astype(float)
  df.High = df.High.astype(float)
  df.Low = df.Low.astype(float)

  print(df)
  

  return df

def main(argv):

  nemo = "LTM.SN"
  interval = "5min"

  #data = getDailyData(nemo, 'full')
  #data = getIntradayData(nemo, interval)

  data = getDailyDataFromMarketstack()

  data.head()
  data.info()
  data.describe()

  min_date = min(data['Date'])
  max_date = max(data['Date'])
  training_years = 3

  data = data[["Date","Close"]]

  # Rename the features: These names are NEEDED for the model fitting
  data = data.rename(columns = {"Date":"ds","Close":"y"}) #renaming the columns of the dataset
  data.head(5)  

  m = Prophet(changepoint_range=1) # the Prophet class (model)

  m.fit(data) # fit the model using all data

  changepoints = m.changepoints
  print(changepoints)


  future = m.make_future_dataframe(periods=5) #we need to specify the number of days in future
  prediction = m.predict(future)

  figure = m.plot(prediction)
  a = add_changepoints_to_plot(figure.gca(), m, prediction)
  plt.title("Prediction of the LTM Price using the Prophet")
  plt.xlabel("Date")
  plt.ylabel("Close Stock Price")
  plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])