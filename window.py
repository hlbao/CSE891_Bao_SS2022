import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
# train ==  (7462, ) last col 
dataset_train = np.loadtxt("train.csv",skiprows=1)
#this is the original data set from Kaggle
dataset_test = np.loadtxt("test.csv",skiprows=1)

# window size
ws = 100;

def windowize( dataset, ws ):
	recs, features = dataset.shape
	signals = np.ndarray(shape=(recs-ws,features-1,ws), dtype=float, order='F')
	# we want to compute (samplesize-ws) rectangular overlapping windows. 
	# this means we can not make predictions on the first ws records
	for i in range(0,recs-ws):
		signals[i, :, :] = dataset[ i:i+ws, :-1].transpose()
	labels = dataset[ws:,-1].astype(float)
	return (signals , labels)
train_x, train_y = windowize(dataset_train, ws)
test_x, test_y = windowize(dataset_test, ws)
plt.show( plt.imshow(train_x[100,:,:],interpolation="none") )
