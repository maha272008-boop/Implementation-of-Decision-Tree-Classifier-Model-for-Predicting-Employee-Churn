# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and create the employee dataset for analysis.

2.Convert the categorical data into numerical format and separate the input features and output values.

3.Split the dataset into training data and testing data for model training and evaluation.

4.Train the Decision Tree Classifier model using the training dataset.

5.Predict the employee results, evaluate the model using accuracy and confusion matrix, and display the decision tree graph.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: MAHALAKSHMI P
RegisterNumber:  212225100024
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import tree
import matplotlib.pyplot as plt
data = {
    'satisfaction_level': [0.38, 0.80, 0.11, 0.72, 0.37, 0.41, 0.10, 0.92],
    'last_evaluation': [0.53, 0.86, 0.88, 0.87, 0.52, 0.50, 0.77, 0.89],
    'number_project': [2, 5, 7, 5, 2, 2, 6, 5],
    'average_monthly_hours': [157, 262, 272, 223, 159, 153, 247, 224],
    'time_spend_company': [3, 6, 4, 5, 3, 3, 4, 5],
    'Work_accident': [0, 0, 0, 0, 0, 0, 0, 0],
    'promotion_last_5years': [0, 0, 0, 0, 0, 0, 0, 0],
    'Departments': ['sales','accounting','hr','technical','support','management','marketing','product'],
    'salary': ['low','medium','medium','high','low','low','medium','high'],
    'left': [1, 1, 1, 1, 1, 0, 1, 0]  
}
df = pd.DataFrame(data)
df = pd.get_dummies(df, columns=['Departments','salary'], drop_first=True)
X = df.drop('left', axis=1)
y = df['left']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
plt.figure(figsize=(20,10))
tree.plot_tree(dt_model, feature_names=X.columns, class_names=['Stayed','Churn'], filled=True)
plt.show()

```

## Output:
<img width="747" height="629" alt="{5C25F41F-F7BB-4280-B919-36C72AE0267D}" src="https://github.com/user-attachments/assets/32e4c8c8-aa87-4f8a-ac8a-692192e753d9" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
