# trainWithNull
ML

# method 1
Created a dataset with filling missing (null) values with zero by not changing the original dataset.
dataset1 = dataset.fillna(0)
# method2
Filling missing values with the next ones.  
dataset2 = dataset.fillna(method ='bfill')
# method 3
Created a new dataset with dropping null values. (since dataset gets smaller, this is not desirable with little data).
dataset3 = dataset.dropna()
# method 4
Filling the null values with the mean of the values from that column for each class.
dataset4 = (dataset.fillna(dataset.groupby('RiskLevel').transform('mean')).astype(int))
