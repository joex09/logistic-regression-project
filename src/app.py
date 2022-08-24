#imports
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

#load and read
df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv', sep=";")

#cleanning data
df = df[df.duplicated() == False]
df['job'] = pd.Categorical(df['job'])
df['marital'] = pd.Categorical(df['marital'])
df['housing'] = pd.Categorical(df['housing'])

#duplicating data
df_copy = df.copy()
df_copy['age'] = pd.cut(x=df_copy['age'], bins=[10,20,30,40,50,60,70,80,90,100])
df_copy = df_copy.replace({'no': 0, 'yes': 1})

#transformation
df_copy = pd.get_dummies(df_copy, columns = ['job', 'marital', 'default','housing', 'loan', 'contact', 'poutcome'])

month_dict={'may':5,'jul':7,'aug':8,'jun':6,'nov':11,'apr':4,'oct':10,'sep':9,'mar':3,'dec':12}
df_copy['month']= df_copy['month'].map(month_dict) 

day_dict={'thu':5,'mon':2,'wed':4,'tue':3,'fri':6}
df_copy['day_of_week']= df_copy['day_of_week'].map(day_dict) 

encoder = LabelEncoder()

df_copy['age'] = encoder.fit_transform(df_copy['age'])
df_copy['education'] = encoder.fit_transform(df_copy['education'])

X=df_copy.drop('y', axis=1)
y=df_copy['y']
X_dummies = pd.get_dummies(X, drop_first= True)

#scaler
scaler = StandardScaler()
scaler.fit(X_dummies)
X_dummies = scaler.transform(X_dummies)

X_train,X_test,y_train,y_test=train_test_split(X_dummies,y,random_state=40, test_size= 0.25)

#modeling
model = LogisticRegression(random_state=1234)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# define second model and parameters

model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]

# define grid search

grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_dummies, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

model2 = LogisticRegression(C= 0.1, penalty= 'l2', solver= 'lbfgs', random_state=132, class_weight={0:0.35, 1:0.65})
model2.fit(X_train, y_train)