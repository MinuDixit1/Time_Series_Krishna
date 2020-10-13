import pandas as pd
import numpy as np

#from Minu_Timeseries import timeseries_withdrawl
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.layers import LSTM
from keras  import callbacks
from keras import optimizers
import tensorflow as tf
import pickle
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils


def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model

# Hotfix function
def make_keras_picklable():

    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__

# Run the function
make_keras_picklable()


def timeseries_withdrawl(Excel_Name, Sheet_Name) :
    df = pd.read_excel(Excel_Name,sheet_name = Sheet_Name , index_col=0)

    #Dropping shitty Columns
    df = df.drop(['CHQ.NO.','VALUE DATE','.'],axis = 1)

    #Removing 'Account No' from index
    df['Account No'] = df.index
    df.reset_index(drop=True, inplace = True)

    #Converting Nan to 0s
    df['WITHDRAWAL AMT'] = df['WITHDRAWAL AMT'].fillna(0)
    df['DEPOSIT AMT'] = df['DEPOSIT AMT'].fillna(0)

    #Creating a new dataframe, with a column for Withdrawl Amount for each day.
    df.sort_values(by=['DATE'])
    day_1 = df['DATE'][0]
    day_n = df['DATE'][len(df)-1]
   
    df_1 = pd.DataFrame({'Date': pd.date_range(start = day_1, end = day_n), 'Withdrawl_Amount' : 0})
    d = df['DATE'][0]
    final_amount = 0

    for i in df.index: 
        dt = df['DATE'][i]
        amt = df['WITHDRAWAL AMT'][i] 
    
        if(dt == d) : 
            final_amount = final_amount + amt
        else : 
            df_1.loc[df_1['Date'] == d, ['Withdrawl_Amount']] = final_amount
            final_amount = df['WITHDRAWAL AMT'][i] 
    
        d = dt  
    
    #Handling the last row
    df_1.loc[df_1['Date'] == d, ['Withdrawl_Amount']] = final_amount
    
    #Setting Date as Index
    df_1 = df_1.set_index('Date')
    
    return df_1


Excel_Name = 'ac_statement.xlsx'
Sheet_Name = 'DS_1'
df1 = timeseries_withdrawl(Excel_Name, Sheet_Name)

df1.plot()


#Scaling the Data between 0 to 1 :
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df1)

#Create the model
train_size = int(len(scaled) * 0.70)
test_size = len(scaled - train_size)
train, test = scaled[0:train_size, :], scaled[train_size: len(scaled), :]

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 25
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)


# LSTM demands the formatting of the data in (Sample, #Features, Window_Size)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


#The network has a visible layer with 1 input, a hidden layer with 4 LSTM blocks or neurons, 
# and an output layer that makes a single value prediction. 
# The default sigmoid activation function is used for the LSTM blocks. 
# The network is trained for 100 epochs and a batch size of 1 is used.
#create and fit the LSTM network

batch_size = 1 #How many records we are pushing into the model
model = tf.keras.Sequential() #Defining Sequential Model

# 4 --> # Total #Neurons 
# (batch_size, look_back, 1) --> 1-D data we have passed, as there is only #passengers feature
# stateful=True means we are allowing the LSTM to hold the information nad forward the information to the next
model.add(LSTM(100, batch_input_shape=(batch_size, look_back, 1), stateful=True))

#Output Layer
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=5, batch_size=batch_size, verbose=2, shuffle=True)


# Saving model to disk wb means write bytes mode
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

import math
from sklearn.metrics import mean_squared_error

#Prediction done on Training Data
trainPredict = model.predict(X_train, batch_size=batch_size)
model.reset_states()

#Prediction done on Testing Data
testPredict = model.predict(X_test, batch_size=batch_size)

# inversing MinMaxScaler for both predicted value and actual value
trainPredict = scaler.inverse_transform(trainPredict)
y_train = scaler.inverse_transform([y_train])

testPredict = scaler.inverse_transform(testPredict)
y_test = scaler.inverse_transform([y_test])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

#print('=============================y_test=======================================')
#print(y_test)

#print('=============================testPredict==================================')
#print(testPredict)

'''
trainPredictPlot = np.empty_like(scaled)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(scaled)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(scaled)-1, :] = testPredict
# plot baseline and predictions
plt.figure(figsize=(20,10))
plt.plot(scaler.inverse_transform(scaled))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
'''

