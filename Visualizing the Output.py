# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('Outputs.csv')
print(dataset)
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1:4].values

# Visualising the Outputs (results)
sns.lineplot(X, y[:,0])
plt.scatter(X, y[:,0])
plt.title('Accuracy')
plt.xlabel('Modules')
plt.ylabel('Accuracy')
plt.xticks(rotation=60)
plt.show()

# Visualising the Outputs (results)
sns.lineplot(X, y[:,1])
plt.scatter(X, y[:,1])
plt.title('Standard Deviation')
plt.xlabel('Modules')
plt.ylabel('Standard Deviation')
plt.xticks(rotation=60)
plt.show()

# Visualising the Outputs (results)
sns.lineplot(X, y[:,2])
plt.scatter(X, y[:,2])
plt.title('Test-Set Accuracy Score')
plt.xlabel('Modules')
plt.ylabel('Test-Set Accuracy Score')
plt.xticks(rotation=60)
plt.show()

# Print Performances and Summary 
print('\n Performances:\n',dataset)
summary1 = {'':['Minimum','Maximum'],
              'Accuracy':[y[:-1,0].min(), y[:-1,0].max()],
              'Module_Name':[X[:-1][list(y[:-1,0]).index(y[:-1,0].min())], X[:-1][list(y[:-1,0]).index(y[:-1,0].max())]]}
summary1 = pd.DataFrame(summary1, index=None)
print('\n Summary:\n',summary1)
summary2 = {'':['Minimum','Maximum'],
             'Standard Deviation':[y[:-1,1].min(), y[:-1,1].max()],
             'Module_Name':[X[:-1][list(y[:-1,1]).index(y[:-1,1].min())], X[:-1][list(y[:-1,1]).index(y[:-1,1].max())]]}
summary2 = pd.DataFrame(summary2, index=None)
print('\n Summary:\n',summary2)
summary3 = {'':['Minimum','Maximum'],
             'Test-Set Accuracy Score':[y[:,2].min(), y[:,2].max()],
             'Module_Name':[X[list(y[:,2]).index(y[:,2].min())], X[list(y[:,2]).index(y[:,2].max())]]}
summary3 = pd.DataFrame(summary3, index=None)
print('\n Summary:\n',summary3)