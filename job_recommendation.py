

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

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#transformed_X = transformer.fit_transform() 
def encoding_train(X_new):
  from sklearn.compose import ColumnTransformer
  from sklearn.preprocessing import OneHotEncoder
  ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
  X_new = np.array(ct.fit_transform(X_new))
  #print(X_new[0])
  #print(X_new.shape)
  ct_1 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [12])], remainder='passthrough')
  X_new= np.array(ct_1.fit_transform(X_new))
  #print(X_new[:17:])
  return X_new

def encoding_test(X_new):
  from sklearn.compose import ColumnTransformer
  from sklearn.preprocessing import OneHotEncoder
  ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
  X_new = np.array(ct.fit_transform(X_new))
  #print(X_new[0])
  #print(X_new.shape)
  ct_1 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [13])], remainder='passthrough')
  X_new= np.array(ct_1.fit_transform(X_new))
  #print(X_new[:17:])
  return X_new

#print(X[0])

"""## Splitting the dataset into the Training set and Test set"""

#print(y_train)

#print(X_train)

#print(X_test)

#print(y_test)

#print(X[0][19:23])

"""## Feature Scaling"""

def scaling(X):
  from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()
  X[:, 17:] = sc.fit_transform(X[:, 17:])
  return X
  #X_test[:, 19:] = sc.transform(X_test[:, 19:])

def pre_processing_train(X):
    X=encoding_train(X)
    #print(X[0])
    X=scaling(X)
    return X

def pre_processing_test(X):
    X=encoding_test(X)
    #print(X[0])
    X=scaling(X)
    return X

#print(X_train)

#print(X_test)

X_train=pre_processing_train(X_train)

#print(X_train)

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

"""## Training the Decision Tree Classification model on the Training set"""

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

"""## Predicting a new result

## Predicting the Test set results
"""

def predict(X_test):
  y_pred = classifier.predict(X_test)
  return (le.inverse_transform(y_pred))
  #print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

"""## Making the Confusion Matrix"""

X_test_2=X_test
#print(X_test[0])

#data = [[10, 8,7,'Technology','Texas',98000]]
def output(age,career_gap,experience,domain,location,salary):
  
  X_test=X_test_2
  print(X_test)
  print(type(domain))
  new_row = np.array([float(age), career_gap,experience,domain,location,salary])
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

x = float(10.9)

answer = output(float(10.9),float(8.9),float(7.8),'Technology','Texas',float(100000.0))
print(answer)

#X_new= encoding(X_test)
#print(X_new)
def ans():
  x = float('10.9')
  answer = output(x,8.9,7.8,'Technology','Texas',100000.0)
  print(answer)
  return ans


"""


cat_features = ['x1','x2']
mmp_filled[cat_features] = mmp_filled[cat_features].astype(str)

one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot", 
                                  one_hot,
                                  ['x1','x2'])],
                                  remainder="passthrough")
"""