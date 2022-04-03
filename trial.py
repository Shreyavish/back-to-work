

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""## Importing the dataset"""

url = "https://raw.githubusercontent.com/bmounikareddy98/Machine-learning-assignments/main/Job_recommendation.csv?token=GHSAT0AAAAAABTE75RGZET4KXX4POU2IJ3WYSIZTQA"
dataset = pd.read_csv(url)
dataset=dataset.dropna()
print(dataset.head())
print(dataset.shape)

X = dataset.iloc[:, 1:7].values
y = dataset.iloc[:, 7].values
#print(X)
#print(y)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
X[:, 3] = le.fit_transform(X[:,3])
X[:, 4] = le.fit_transform(X[:,4])
print("enter")
#transformed_X = transformer.fit_transform() 
def encoding_train(X_new):
  from sklearn.compose import ColumnTransformer
  from sklearn.preprocessing import OneHotEncoder
  ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder())])
  X_new = np.array(ct.fit_transform(X_new))
  return X_new

def scaling(X):
  from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()
  X[:, 1:7] = sc.fit_transform(X[:, 1:7])
  return X

def pre_processing_train(X):
    X=encoding_train(X)
    #print(X[0])
    X=scaling(X)
    return X
pre_processing_train(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)


def predict(X_test):
  y_pred = classifier.predict(X_test)
  return (le.inverse_transform(y_pred))

X_test_2=X_test
#print(X_test[0])

#data = [[10, 8,7,'Technology','Texas',98000]]
def output(age,career_gap,experience,domain,location,salary):
  
  X_test=X_test_2
  print(X_test)
  new_row = np.array([age, career_gap,experience,domain,location,salary])
  X_test = np.append(X_test,[new_row],axis= 0)
  print(X_test)
  X_test= pre_processing_test(X_test)
  new_column = []
  X_test = np.delete(X_test, 1, 1)
  #an_array = np.insert(an_array, 1, new_column, axis=1)
  res=predict(X_test)
  #print(res.tail())
  result=res[-1]   
  #print(result)
  #print(X_test[-1:])
  X_test=X_test_2
  return result


answer = output(10.9,8.9,7.8,'Technology','Texas',100000.0)
print(answer)
#X_new= encoding(X_test)
#print(X_new)
def ans():
  answer = output(10.9,8.9,7.8,'Technology','Texas',100000.0)
  print(answer)
  return ans