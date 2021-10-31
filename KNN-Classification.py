import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from sklearn import preprocessing

# !wget -O teleCust1000t.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv

df = pd.read_csv('teleCust1000t.csv')
print("\n Csv File Overview: \n")
print(df.head())

# Feature Set

print("\n Featuers: \n")
print(df.columns)

#To use scikit-learn library , we have to convert Pandas data frame to a Numpy array:

X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
print("\n Values of Featuers: \n")
print(X[0:5])

y = df['custcat'].values
print("\n Values of Classifier: \n")
print(y[0:5])

# Data Normalization

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X.astype(float))
print("\n Normalized Data: \n")
print(X[0:5])
print("\n")


# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
print("\n")

# Classification

from sklearn.neighbors import KNeighborsClassifier

# Training - k=4

k = 4
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
print(neigh)

# Predict

yhat = neigh.predict(X_test)
print("Y_hat: " , yhat[0:5]) 
print("Y    : " , y[0:5])
print("\n")

# Accuracy evlauation

from sklearn import metrics

print("Train set Accuracy: ", metrics.accuracy_score(y_train , neigh.predict(X_train)))
print("Test set Accuracy: " , metrics.accuracy_score(y_test , yhat))
print("\n")

# Training - k = 6

k = 6
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
print(neigh)

# Predict

yhat = neigh.predict(X_test)
print("Y_hat: " , yhat[0:5]) 
print("Y    : " , y[0:5])
print("\n")

# Accuracy evlauation

print("Train set Accuracy: ", metrics.accuracy_score(y_train , neigh.predict(X_train)))
print("Test set Accuracy: " , metrics.accuracy_score(y_test , yhat))
print("\n")

# Find the best K

Ks = 10

mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

print("Calculate y_hats for each K: ")
print("Ys : " ,y[0:5])
print("\n")


print("Yhats :")

for k in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[k-1] = metrics.accuracy_score(y_test, yhat)
    print("k = ",k,"Y_hat:", yhat[0:5] , "Accuracy:" , mean_acc[k-1])    
    std_acc[k-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

print("\n")

print("Accuracy Plot:\n")

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
print("Plot will Open ...")
plt.show()
print("Plot has Closed ...")

print( "\nThe best accuracy was with", mean_acc.max(), "with k =", mean_acc.argmax()+1)




