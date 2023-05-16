import numpy as np 
import pandas as pd 
from tensorflow import keras
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from keras.callbacks import ModelCheckpoint
import plotly.graph_objs as go
import plotly.offline as pyo

# Loading the model
model = keras.models.load_model("LSTM_model.h5")

#Scaling test data between 0 and 1
sc = MinMaxScaler(feature_range=(0,1))
timestep = 60
# cleaning and Reading the test data 
test = input('Enter the name of the data set to test on: ') +'.csv'

# clean_csv(test)
testData = pd.read_csv('./Testing/'+test)
date = testData['Date']
# Making the Close column into a number Then cleaning the dataset using the clean function ceated at the beginnig 
testData["Close"]=pd.to_numeric(testData.Close,errors='coerce')

data = pd.read_csv('combined_data.csv')
data["Close"]=pd.to_numeric(data.Close,errors='coerce')
trainData = data.iloc[:,4:5].values

# MinMacScaler is used to scale our data between 0 and 1

# Selecting the close column and then taking the values
testData = testData.iloc[:,4:5]
print(testData)
y_test = testData.iloc[timestep:,0:].values

inputClosing = testData.iloc[:,0:].values
data_max = data['Close'].max()
test_max = testData['Close'].max()
if(data_max>=test_max):
  trainData = sc.fit_transform(trainData)
else:  
  trainData = sc.fit_transform(testData)


inputClosing_scaled = sc.transform(inputClosing)
X_test = []
length = len(testData)

#Appending the X test array using the scaled version of inputClosing and taking into consideration the timestep

for i in range(timestep,length): 
    X_test.append(inputClosing_scaled[i-timestep:i,0])
X_test = np.array(X_test)
print(X_test)

#Reshape of X_test since LSTM takes on 3D arrays instead of 2D
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

print(X_test)
#predicting
y_pred = model.predict(X_test) 

# Inverse transforming the predictions
predicted_price = sc.inverse_transform(y_pred)
# print(predicted_price)
# Plotting the predicted and the actual stock prices
print(y_test)
# plt.plot(y_test, color = 'red', label = 'Actual Stock Price')
# plt.plot(predicted_price, color = 'green', label = 'Predicted Stock Price')
# plt.title('Google stock price prediction')
# plt.xlabel('Time')
# plt.ylabel('Stock Price')
# plt.legend()
# plt.show()


# Flatten y_test and predicted_price
y_test = np.array(y_test).flatten()
predicted_price = np.array(predicted_price).flatten()

# Create the traces
trace_actual = go.Scatter(x = date, y = y_test, mode = 'lines', name = 'Actual Stock Price', line = {'color': 'red'})
trace_predicted = go.Scatter(x = date, y = predicted_price, mode = 'lines', name = 'Predicted Stock Price', line = {'color': 'green'})

# Define the layout
layout = go.Layout(title = test[0:len(test)-4]+' stock price prediction', xaxis = {'title': 'Date'}, yaxis = {'title': 'Stock Price'})

# Create the figure object
fig = go.Figure(data = [trace_actual, trace_predicted], layout = layout)

# Display the plot
pyo.iplot(fig)
