import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
In [60]:
df = pd.read_csv(C:/Users/HP/Downloads/churn_modelling.csv')
In [61]:
#Separate the independent variables (features) from the dependent variable (label):
X = df.iloc[:, 3:13].values
y = df.iloc[:, 13].values
In [62]:
#Encode categorical variables:
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

X = X[:, 1:]
In [63]:
#Split the dataset into training and testing sets:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
In [64]:
#Create a random forest classifier object:
from sklearn.ensemble import RandomForestRegressor
classifier = RandomForestRegressor(n_estimators = 100, criterion = 'entropy', random_state = 0)
In [ ]:
#Fit the model on the training data:
classifier.fit(X_train, y_train)
In [ ]:
#Predict the labels on the testing set:
y_pred = classifier.predict(X_test)
In [ ]:
#Evaluate the model's performance:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}%'.format(accuracy*100))
In [ ]:
 
