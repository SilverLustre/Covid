import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
import pandas as pd
from keras.layers import Embedding
import datetime
import seaborn as sns


# convert an array of values into a dataset matrix

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


# fix random seed for reproducibility
numpy.random.seed(7)
# Read csv file
dateparse = lambda x: pd.datetime.strptime(x, '%Y/%m/%d')
dataframe = pd.read_csv('test.csv', parse_dates=['Date'], date_parser=dateparse)
# Sort dataset by column Date
dataframe = dataframe.sort_values('Date')
dataframe = dataframe.groupby('Date')['Total'].sum().reset_index()
dataframe.set_index('Date', inplace=True)
dataframe.head()
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
print(train_size)
test_size = len(dataset) - train_size
print(test_size)
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
# reshape into X=t and Y=t+1
look_back = 90
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network

model = Sequential()
model.add(LSTM(units=80, return_sequences=True, input_shape=(1, look_back)))
model.add(Dropout(0.1))
model.add(LSTM(units=80, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(units=80))
model.add(Dropout(0.05))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5)
history = model.fit(trainX, trainY, epochs=200, batch_size=15, validation_data=(testX, testY), callbacks=[reduce_lr],
                    shuffle=False)

# ---------------make predictions------------------------------------------
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
# plot baseline and predictions
plt.figure(figsize=(18, 10))
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

'''print(testX.shape)
print(testY.shape)
print(trainX.shape)
print(trainY.shape)
import numpy as np
np.save(file="./testX_1LSTM.npy", arr=testX)
np.save(file="./testY_1LSTM.npy", arr=testY)
np.save(file="./trainX_1LSTM.npy", arr=trainX)
np.save(file="./trainY_1LSTM.npy", arr=trainY)
'''
# reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=5)

# history=model.fit(trainX, trainY, epochs = 200, batch_size = 15,validation_data=(testX, testY), callbacks=[reduce_lr],shuffle=False)
