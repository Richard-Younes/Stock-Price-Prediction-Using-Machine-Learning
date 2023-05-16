import numpy as np 
import pandas as pd 
from tensorflow import keras
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from keras.callbacks import ModelCheckpoint

# Here csv_name represents the dataset we want to train the model with.

csv_name = input("Enter the name of the csv file: ") +'.csv'

# The following function is used to clean the dataset by checking for NAN values, and if its less than or equal than 5% it drops them,
# else if the NAN are more than 5% the function replace them by the mean, of the specific column.
def clean_csv(csv_name):
    # Loading the dataset
    df = pd.read_csv(csv_name)

    # Calculate the percentage of missing values in each column.
    percent_missing = df.isna().sum() / len(df) * 100
    print('percent_missing: ', percent_missing)
    
    # Iterate through the columns and handle missing values.
    for col in df.columns:
        if percent_missing[col] <= 5:
            # Drop rows with missing values.
            df = df.dropna(subset=[col])
        else:
            # Replace missing values with the column mean.
            mean = np.mean(df[col])
            df[col] = df[col].fillna(mean)
    
    # Export the cleaned DataFrame to the old csv.
    df.to_csv(csv_name, index=False)

# Calling the finction that clears the DataFrame.
clean_csv(csv_name)

# You can comment the above code if the data is clean or don't want to clean it

# Here we took the clean csv and read it to the variable data then printing the head and info just to get additional information.
data = pd.read_csv(csv_name)
print(data.head())
print(data.info())

# We are making sure that the close column is numeric type and making all data from column into an array.
data["Close"]=pd.to_numeric(data.Close,errors='coerce')
trainData = data.iloc[:,4:5].values

# MinMacScaler is used to scale our data between 0 and 1
sc = MinMaxScaler(feature_range=(0,1))
trainData = sc.fit_transform(trainData)
print('Traned data Shape: ',trainData.shape)

# We create the X_train and Y_train which are 2 empty lists and the timestep which we chose its value to be 60
X_train = []
y_train = []
timestep = 60

# The for loop is used to fill the 2 lists while taking the timestep into account
for i in range (timestep,len(trainData)): 
    X_train.append(trainData[i-timestep:i,0]) 
    y_train.append(trainData[i,0])
    
# Transforming the lists to array
X_train,y_train = np.array(X_train),np.array(y_train)

# Reshaping X_train since the LSTM model needs the array to be 3D instead of 2D
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1)) 

print('The shape of X_train after the reshape is: ',X_train.shape)


# Creating the sequential model
model = Sequential()

# Adding the Long Short Term Memory (LSTM) layers.
model.add(LSTM(units=100, return_sequences = True, input_shape =(X_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units=20, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences = False))
model.add(Dropout(0.2))

model.add(Dense(units =1))

# Compiling the model using the adam optimizer and the mean_sqaured_error loss function
model.compile(optimizer='adam',loss="mean_squared_error", metrics=["accuracy"])

# Fitting the mode
hist = model.fit(X_train, y_train, epochs = 20, batch_size = 32, verbose=1, validation_split=0.3)


# Plotting the loss function in respect with the number of Epochs
plt.plot(hist.history['loss'])
plt.title('Training model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# Saving the data Model
model.save("my_model.h5")