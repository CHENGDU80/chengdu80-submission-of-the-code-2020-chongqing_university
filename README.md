# Chongqing University Solution for Chengdu 80 Competition 

## Overview
This is the solution code repository of Chongqing University to Chengdu 80 competition. 
The code repository can be divided into two parts: 
algorithm design and website display.

### Algorithm Design
In the competition, we implemented three algorithms to predict stock prices: 
**ARIMA**, **LSTM-Based PTS**, and a **highly interpretable Logistic regression model** 
derived from knowledge distillation of LSTM. 

#### Data Preprocessing
- News Sentiment. News sentiment expresses the mainstream media’s positive or negative predictions 
about the company, which is subjective. We use the method based on 
professional financial news dictionary (Loughran and McDonald Financial Sentiment Dictionaries) to get news attitude.
- Technical Indicators. We extract 10 technical indicators: 
    "MA10", "MA20", "MA30", "DIFF", "DEA", "MACD", "RSI6", "RSI12", "RSI24", "MFI".
    
| Name | Description |
| ---- | ------ |
|MA10 | 10-day close price moving average |
|MA20 | 20-day close price moving average |
|MA30 | 30-day close price moving average |
| DIFF | The difference between EMA12 and EMA26  |
| DEA | 9-day exponential moving average of DIFF  |
|MACD | Moving average convergence and divergence|
| RSI6| 6-day relative strength index |
|RSI12 | 12-day relative strength index |
|RSI24 | 24-day relative strength index |
|MFI| Money flow index|

#### ARIMA Model

ARIMA is one of the most classic and most widely used statistical forecasting techniques when dealing with univariate time series. It basically uses the lag values and lagged forecast errors to predict the feature values.

ARIMA model has three parameters including:

• **p**: The number of lag observations included in the model, also called the lag order.

• **d**: The number of times that the raw observations are differenced, also called the degree of differencing.

• **q**: The size of the moving average window, also called the order of moving average.

#### LSTM-based PTS Model

We use three characteristic indexes including original data, technical indicators and sentimental analyze. 

About technical indicators you can see it in the table above.

About sentimental analyze, we extract every word's polarity and subjectivity separately in the news based on Harvard IV-4 and Loughran and McDonald Financial Sentiment Dictionaries.(About these dictionaries, you can get more information from the paper: Information Processing and Management).We calculate POS and NEG by adding words’ polarity and subjectivity separately

By using the formula behind, we  calculate the final value

$Polarity = \frac{POS-NEG}{POS+NEG}$

$Subjectivity = \frac{POS+NEG}{count(*)}$

We merge both technical indicators data and sentiment analyze data to the original data

And then we use a three layers LSTM neutral network and 1 layer fully connected layer

#### Knowledge Distillation

By using a A single hidden layer neural network with 17 feature inputs and 1 feature output, we get the weight of each characteristic index

![image-20201029194637550](/Users/songxinyi/Library/Application Support/typora-user-images/image-20201029194637550.png)

### Web Interface

Based on flask framework, Javascript and HTML5, we implent the following functions:

* Look up companys
* Show historical data
* Show prediction data
* Show daily news
* Show the relation between companys
* Show the result of Knowledge Distillation in order to explain our prediction







