import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from plotly.offline import plot, iplot
import plotly.graph_objs as go

x=input("Enter the name of the Data set you would like to predict:")
x=x+'.csv'

df=pd.read_csv('./Testing/',x)
print(df.head())

df["Date"]=pd.to_datetime(df["Date"])


print("First date: " ,df.Date.min())
print("Last Date: ", df.Date.max())


print("Total Days= ", df.Date.max()-df.Date.min())

print(df.describe())

df[["Open", "High","Low","Close", "Adj Close"]].boxplot()
plt.show()

plt.plot(df["Date"], df["Close"])
plt.show()



#Building regression model
from sklearn.model_selection import train_test_split

#for processing
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#for Model evaluation
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

#Split data into train and test 

X=np.array(df.index).reshape(-1,1)
y=df["Close"].values

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)

scaler= StandardScaler().fit(X_train)

from sklearn.linear_model import LinearRegression

#Creating linear model

ln= LinearRegression()

ln.fit(X_train,y_train)

layout= go.Layout(
    title="Stock Prices",
    xaxis=dict(
        title="Date",
        titlefont=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    ),
    yaxis= dict(
        title="Price",
        titlefont=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    
)

trace0=go.Scatter(
    x=X_train.T[0],
    y=y_train,
    mode="markers",
    name="Actual"
)

trace1= go.Scatter(
    x=X_train.T[0],
    y=ln.predict(X_train).T,
    mode="lines",
    name="Predicted"    
)

Data=[trace0,trace1]
layout.xaxis.title.text="Day"

plot2=go.Figure(data=Data,layout=layout)

iplot(plot2)

#Calcule the score for model Evaluation

scores=f'''
{"Metric".ljust(10)}{"Train".center(20)}{"Test".center(20)}
{"r2_score".ljust(10)}{r2_score(y_train,ln.predict(X_train))}\t {r2_score(y_test,ln.predict(X_test))}
{"MSE".ljust(10)}{mse(y_train, ln.predict(X_train))}\t {mse(y_test,ln.predict(X_test))}    
'''
print(scores)
