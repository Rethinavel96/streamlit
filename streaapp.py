import streamlit as slt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('iris.csv')
print(df.head(4))
print(df.isna().sum())
df.drop('Id',inplace=True,axis=1)


df['Species']=LabelEncoder().fit_transform(df['Species'])
print(df)

rfc=RandomForestClassifier(n_estimators=50,criterion='entropy')

p={'criterion':['gini','entropy'],'n_estimators':[50,100,150,200]}

g=GridSearchCV(rfc,param_grid=p)

x=df.drop('Species',axis=1)
y=df['Species']
g.fit(x,y)

print(g.best_score_)
print(g.best_params_)

rfc.fit(x,y)





slt.title('Welcome to the iris flower type prediction model by Yogesh')

a=slt.number_input('Enter the sepal length')
b=slt.number_input('Enter the sepal width')
c=slt.number_input('Enter the petal length')
d=slt.number_input('Enter the petal width')

result=rfc.predict([[a,b,c,d]])
if result==[0]:
    result='The flower model is Iris-setosa'
elif result==[1]:
    result='The flower model is Iris-versicolor'
else:
    result='The flower model is Iris-virginica'


if slt.button('click here to find iris model'):
    slt.success(result)
    slt.balloons
primaryColor="#F63366"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#262730"
font="sans serif"




