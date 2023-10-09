# HEART_DIESEASE_DATASET
HEART_DIESEASE_DATASET  DATA VISUALISATION USING PYTHON


import pandas as pd
a=pd.read_csv("heart.csv")
a.isna()

from sklearn import preprocessing
y=preprocessing.LabelEncoder()
a["AGE_50"]=y.fit_transform(a["AGE_50"])
a["MD_50"]=y.fit_transform(a["MD_50"])
a["SBP_50"]=y.fit_transform(a["SBP_50"])
a["DBP_50"]=y.fit_transform(a["DBP_50"])
a["HT_50"]=y.fit_transform(a["HT_50"])
a["WT_50"]=y.fit_transform(a["WT_50"])
a["CHOL_50"]=y.fit_transform(a["CHOL_50"])
a["SES"]=y.fit_transform(a["SES"])
a["CL_STATUS"]=y.fit_transform(a["CL_STATUS"])
a["MD_62"]=y.fit_transform(a["MD_62"])
a["SBP_62"]=y.fit_transform(a["SBP_62"])
a["DBP_62"]=y.fit_transform(a["DBP_62"])
a["CHOL_62"]=y.fit_transform(a["CHOL_62"])
a["WT_62"]=y.fit_transform(a["WT_62"])
a["IHD_DX"]=y.fit_transform(a["IHD_DX"])
a["DEATH"]=y.fit_transform(a["DEATH"])
a["AGE_50"].unique()
a["MD_50"].unique()
a["SBP_50"].unique()
a["DBP_50"].unique()
a["HT_50"].unique()
a["WT_50"].unique()
a["CHOL_50"].unique()
a["SES"].unique()
a["CL_STATUS"].unique()
a["MD_62"].unique()
a["SBP_62"].unique()
a["DBP_62"].unique()
a["CHOL_62"].unique()
a["WT_62"].unique()
a["IHD_DX"].unique()
a["DEATH"].unique()

print("******GRAPHICAL MODEL OF THE GIVEN DATA ******")
import seaborn as sd
sd.boxplot(x="DEATH",y="CHOL_62",data=a)
sd.boxplot(x="DBP_62",y="WT_62",data=a)
sd.boxplot(x="DEATH",y="IHD_DX",data=a)
sd.boxplot(x="DEATH",y="MD_62",data=a)

print("******DISTRIBUSION PLOT******")
sd.distplot(a.DEATH)
sd.distplot(a.DBP_62)
sd.distplot(a.CHOL_62)
sd.distplot(a.IHD_DX)
sd.distplot(a.AGE_50)
sd.distplot(a.MD_50)
sd.distplot(a.WT_50)

print("******BARPLOT******")
sd.barplot(x="AGE_50",y="DBP_50",data=a)
sd.barplot(x="SBP_50",y="WT_50",data=a)
sd.barplot(x="HT_50",y="CHOL_50",data=a)
sd.barplot(x="SES",y="CL_STATUS",data=a)
sd.barplot(x="CHOL_62",y="WT_62",data=a)
sd.barplot(x="IHD_DX",y="DEATH",data=a)

print("******LMPLOT******")
sd.lmplot(x="AGE_50",y="DBP_50",data=a)
sd.lmplot(x="SBP_50",y="WT_50",data=a)
sd.lmplot(x="HT_50",y="CHOL_50",data=a)
sd.lmplot(x="SES",y="CL_STATUS",data=a)
sd.lmplot(x="CHOL_62",y="WT_62",data=a)
sd.lmplot(x="IHD_DX",y="DEATH",data=a)

from matplotlib import pyplot as plt
plt.hist(a["AGE_50"])
plt.hist(a["MD_50"])
plt.hist(a["SBP_50"])
plt.hist(a["DBP_50"])
plt.hist(a["HT_50"])
plt.hist(a["WT_50"])
plt.hist(a["CHOL_50"])
plt.hist(a["WT_62"])

print("*******SCATTER PLOT*******")
from matplotlib import pyplot as plt
plt.scatter(a["AGE_50"],a["MD_50"])
print("******* PLOT GRAPH *******")
from matplotlib import pyplot as plt
plt.plot(a["CHOL_50"],a["MD_50"])

print("******SPLIT TEST AND TRAIN DATA******")
x=a.iloc[:,:-1].values
print(x)
y=a["DEATH"]
print(y)

print("******TRAIN_TEST_SPLIT ALGORITHM******")
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.20,shuffle=True)
print(x_train)
print(x_test)

print("******LOGISTIC REGRESSION ALGORITHM******")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
S=LogisticRegression()
S.fit(x_train,y_train)
predictions=S.predict(x_test)
print("Accuracy score ",accuracy_score(y_test,predictions))

print("******CONFUSION MATRIX******")
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)

print("******CLASSIFICATION REPORT******")
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

print("******PRINTING THE TRAIN DATAS******")
print(x_train)
print(x_test)

print("******RANDOM FOREST CLASIFER ALGORITHM******")
from sklearn.ensemble import RandomForestClassifier
S=RandomForestClassifier()
S.fit(x_train,y_train)
predictions1=S.predict(x_test)
print("Accuracy score ",accuracy_score(y_test,predictions1))

print("******CONFUSION MATRIX******")
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions1)

print("******CLASSIFICATION REPORT******")
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions1))

print("******GAUSSIANNB ALGORITHM******")
from sklearn.naive_bayes import GaussianNB
S=GaussianNB()
S.fit(x_train,y_train)
predictions2=S.predict(x_test)
print("Accuracy score ",accuracy_score(y_test,predictions2))

print("******CONFUSION MATRIX******")
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions2)

print("******CLASSIFICATION REPORT******")
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions2))

print("******KNEIGHBOURS CLASSIFIER ALGORITHM*******")
from sklearn.neighbors import KNeighborsClassifier
S=KNeighborsClassifier()
S.fit(x_train,y_train)
predictions3=S.predict(x_test)
print("Accuracy score",accuracy_score(y_test,predictions3))

print("******CONFUSION MATRIX******")
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions3)

print("******CLASSIFICATION REPORT******")
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions3))

print("******SUPPORT VECTOR MACHINE AlGORITHM******")
from sklearn import svm
S=svm.SVC()
S.fit(x_train,y_train)
predictions4=S.predict(x_test)
print("Accuracy score",accuracy_score(y_test,predictions4))

print("******CONFUSION MATRIX******")
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions4)

print("******CLASSIFICATION REPORT******")
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions4))

print("******DECISIONT TREE CLASSIFIER******")
from sklearn.tree import DecisionTreeClassifier
S=DecisionTreeClassifier()
S.fit(x_train,y_train)
predictions5=S.predict(x_test)
print("Accuracy score",accuracy_score(y_test,predictions5))

print("******CONFUSION MATRIX******")
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions5)

print("******CLASSIFICATION REPORT******")
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions5))

from matplotlib import pyplot as plt
import numpy as np
cars = ['MD_50', 'SBP_50', 'AGE_50',
        'CHOL_50', 'CL_STATUS', 'SES']
data = [23, 17, 35, 29, 12, 41]
fig = plt.figure(figsize =(10, 7))
plt.pie(data)
plt.show()


print("The above details shows the differnt plots and accuracy on each data given by a random user and its visualizations has been presented.")
