
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# get dataset
Dataset = pd.read_csv('crew1_leftSeat_DA.csv')
Dataset.drop(['Unnamed: 0'], axis=1, inplace=True)
Dataset['time'] = Dataset['time'].astype('timedelta64[s]')
Dataset.to_csv('air pilot.csv')

#Data selection
dataset = pd.read_csv('air pilot.csv', index_col='time')
dataset.drop(['Unnamed: 0'], axis=1, inplace=True)
dataset.index.name = 'time'

#parameters check
Eeg_f1 =1 np.array([Dataset.iloc[:,4]])
Ecg = np.array([Dataset.iloc[:,24]])
R = np.array([Dataset.iloc[:,25]])
GSR= np.array([Dataset.iloc[:,26]])
plt.figure(1)
Eeg_f1, = plt.plot(Eeg_f1[0,:])
Ecg, = plt.plot(Ecg[0,:])
R, = plt.plot(R[0,:])
GSR, = plt.plot(GSR[0,:])
plt.legend([Eeg_f1,Ecg,R,GSR], ["Eeg_f1","Ecg","R", "GSR"] )
plt.show()

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
values = dataset.values
labelencoder_1 = LabelEncoder()
values[:, 1] = labelencoder_1.fit_transform(values[:, 1])
labelencoder_2 = LabelEncoder()
values[:, 26] = labelencoder_2.fit_transform(values[:, 26])
values = values.astype('float32')

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
values_scaled = scaler.fit_transform(values)

# Drop columns not to be included in the prediction
Final= pd.DataFrame(values_scaled)
Final.drop([0], axis=1, inplace=True)
Final.drop([1], axis=1, inplace=True)
Final.drop([2], axis=1, inplace=True)

# Splitting  dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X = Final.iloc[:, 3:26].values
y = Final.iloc[:, 23].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0],1, X_train.shape[1]))
X_test = np.reshape(X_test,(X_test.shape[0], 1, X_test.shape[1]))
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#RNN building:
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
# Initialize LSTM
classifier = Sequential()
# Adds first LSTM layer and some Dropout regularisation
classifier.add(LSTM(units = 100, return_sequences = True,
                    input_shape = (X_train.shape[1], X_train.shape[2])))
classifier.add(Dropout(0.3))
# Adds second LSTM layer and some Dropout regularisation
classifier.add(LSTM(units = 100, return_sequences = True))
classifier.add(Dropout(0.4))
# Adds third LSTM layer and some Dropout regularisation
classifier.add(LSTM(units = 100))
classifier.add(Dropout(0.5))
# Adds output layer
classifier.add(Dense(units = 1, activation='sigmoid'))

# Compiles LSTM
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
print(classifier.summary())
# Fits RNN to Training set
history = classifier.fit(X_train, y_train, epochs = 4, batch_size = 1500,
               validation_data=(X_test, y_test))
# Final model evaluation
scores = classifier.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

#scores summaries
from numpy import mean
from numpy import std
def summarize_results(score):
 print(score)
 m, s = mean(score), std(score)
 print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
 
# run an experiment
def run_experiment(repeats=10):
 # repeat experiment
 score = list()
 for r in range(repeats):
  score = scores(X_train, X_test, y_train, y_test)
  score = score * 100.0
  print('>#%d: %.3f' % (r+1, score))
  scores.append(score)
  
# summarize results
    summarize_results(score)
    
# Predictions and visualization:
# Gets prediction for test set
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred1= classifier.predict_proba(X_test)

# invert scaling for forecast
from numpy import concatenate
X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))
inv_ypred = concatenate((y_pred, X_test[:, 1:]), axis=1)
inv_ypred = scaler.inverse_transform(inv_ypred)
inv_ypred = inv_ypred[:,0]

# invert scaling for actual
y_test = y_test.reshape((len(y_test), 1))
inv_y = concatenate((y_test, X_test[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

# Makes the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# results visualization
plt.figure()
plt.plot(y_test,y_pred)
plt.show(block=False)
plt.figure()
test, = plt.plot(y)
predict, = plt.plot(y_pred)
plt.legend([predict,test],["predicted Data","Real Data"])
plt.show()
