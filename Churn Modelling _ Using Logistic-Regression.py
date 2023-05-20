# Importing the libraries
import numpy as np
import pandas as pd

# Data Preprocessing

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

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C= 0.01, max_iter= 100, multi_class= 'ovr', penalty= 'l2', solver= 'newton-cg',random_state = 0)
classifier.fit(X_train, y_train)

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'penalty': ['l1', 'l2', 'elasticnet'], 
                'C': [0.01, 0.1, 1, 10, 100], 
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 
                'max_iter': [100, 500, 1000],
                'multi_class': ['ovr', 'multinomial']}]
grid_search = GridSearchCV(estimator = classifier,
                            param_grid = parameters,
                            scoring = 'accuracy',
                            cv = 10,
                            n_jobs = -1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
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
answer= 'No' if (classifier.predict(sc.transform([customer_])) > 0.5) == 0 else 'Yes'
print(f'Should we say goodbye to this customer {customer} ?', answer)

# Save the output (result) in CSV file :
Header = ['Module_Name', 'Accuracy', 'Standard Deviation', 'Test-Set Accuracy Score']
output= [['Logistic Regression', accuracies.mean()*100, accuracies.std()*100, accuracy_score(y_test, y_pred)]]
output = np.array(output)
Output = pd.DataFrame(output)
try :
    pd.read_csv('C:/Users/Miramin_LPC/Desktop/PythonS/Projects/Churn Modeling/Outputs.csv')
    Output.to_csv('C:/Users/Miramin_LPC/Desktop/PythonS/Projects/Churn Modeling/Outputs.csv', mode='a', index=False, header=False)
except:
    Output.to_csv('C:/Users/Miramin_LPC/Desktop/PythonS/Projects/Churn Modeling/Outputs.csv', mode='w', sep=',', index=False, header=Header)
