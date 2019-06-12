
import pandas as pd
from pandas import read_csv
from matplotlib import pyplot
# load the dataset
data = read_csv('train.csv', header=None,skiprows = 1, usecols=range(4,24))
# retrieve data as numpy array
values = data.values
for i in range(values.shape[1]):
    pyplot.subplot(values.shape[1], 1, i+1)
    pyplot.plot(values[:, i])
pyplot.show()
# remove outliers from the EEG data
from pandas import read_csv
from numpy import mean
from numpy import std
from numpy import delete
from numpy import savetxt
# step over each EEG column
for i in range(values.shape[1] - 1):
# calculate column mean and standard deviation
    data_mean, data_std = mean(values[:,i]), std(values[:,i])
# define outlier bounds
    cut_off = data_std * 4
    lower, upper = data_mean - cut_off, data_mean + cut_off
# remove too small
    too_small = [j for j in range(values.shape[0]) if values[j,i] < lower]
    values = delete(values, too_small, 0)
    print('>deleted %d rows' % len(too_small))
    # remove too large
    too_large = [j for j in range(values.shape[0]) if values[j,i] > upper]
    values = delete(values, too_large, 0)
    print('>deleted %d rows' % len(too_large))

# create a subplot for each time series
pyplot.figure()
for i in range(values.shape[1]):
    pyplot.subplot(values.shape[1], 1, i+1)
    pyplot.plot(values[:, i])
pyplot.show()
pd.DataFrame(values).to_csv("C:/Users/HP/Desktop/train_nooutliers.csv")
