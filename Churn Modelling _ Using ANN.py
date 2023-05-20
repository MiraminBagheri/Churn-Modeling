# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
# print(X)
# print(y)

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
# print(X)

# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
# print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
ann.load_weights('W.h1')
# ann.fit(X_train, y_train, batch_size = 32, epochs = 300)
# ann.save_weights('W.h1')

# Part 4 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix :\n', cm)
print(' Test-Set Accuracy Score: ', accuracy_score(y_test, y_pred))

# Predicting the result of a single observation

"""
Predict if the customer with the following informations will leave the bank: 
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $ 60000
Number of Products: 2
Does this customer have a credit card? Yes
Is this customer an Active Member: Yes
Estimated Salary: $ 50000
So, should we say goodbye to that customer?
"""
customer = [600, 'France', 'Male', 40, 3, 60000, 2, 1, 1, 50000]
customer_=customer.copy()
customer_[2] = le.transform([customer_[2]])[0]
customer_ = ct.transform([customer_])[0]
answer= 'No' if (ann.predict(sc.transform([customer_])) > 0.5) == 0 else 'Yes'
print(f'Should we say goodbye to this customer {customer} ?', answer)

# Save the output (result) in CSV file :
Header = ['Module_Name', 'Accuracy', 'Standard Deviation', 'Test-Set Accuracy Score']
output= [['ANN', '', '', accuracy_score(y_test, y_pred)]]
output = np.array(output)
Output = pd.DataFrame(output)
try :
    pd.read_csv('C:/Users/Miramin_LPC/Desktop/PythonS/Projects/Churn Modeling/Outputs.csv')
    Output.to_csv('C:/Users/Miramin_LPC/Desktop/PythonS/Projects/Churn Modeling/Outputs.csv', mode='a', index=False, header=False)
except:
    Output.to_csv('C:/Users/Miramin_LPC/Desktop/PythonS/Projects/Churn Modeling/Outputs.csv', mode='w', sep=',', index=False, header=Header)
