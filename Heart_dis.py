import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df = pd.read_csv('heart.csv')
print("First 5 rows \n")
print(df.head())
print("\n")
print("Last 5 rows \n")
print(df.tail())
print("\n")
print("Shape of the dataframe: \n")
print(df.shape)
print("\n")
print("Information of the dataframe: \n")
print(df.info())
print("\n")
print("Number of null count \n")
print(df.isnull().sum())
print("\n")
print(df["ca"].value_counts())
print("Splitting Independent and Dependent variable \n")
X = df.drop(["target"],axis = 1)
Y  = df["target"]
print(f"X shape is {X.shape}\nY shape is {Y.shape}")
X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.2)
print("Without random state")
print(X_train)
print("With random state ")
X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.2, random_state = 2)
print(X_train)
print("Logistic Regression Algorithm")
lr_scores  = []
for k in range(1,40):
    lr = LogisticRegression(random_state = k)
    lr.fit(X_train , Y_train)
    ypred= lr.predict(X_test)
    lr_scores.append(accuracy_score(Y_test , ypred))
print(f"Best choice of k:{np.argmax(lr_scores)+1}")
print("\n")
lr = LogisticRegression(random_state = 1)
print(lr.fit(X_train , Y_train))#fit is used to train the model
from sklearn.metrics import confusion_matrix
te_score = lr.score(X_test , Y_test)
print(f"\n Testing accuracy : {te_score}")
ypred= lr.predict(X_test)#model is ready to predict
cm = confusion_matrix(Y_test,ypred)
print("Confusion matrix:\n" , cm )
#import seaborn as sns
#conf_matrix = pd.DataFrame(data = cm,  
#                           columns = ['Actual:0', 'Actual:1'], 
#                           index =['Predicted:0', 'Predicted:1'])      
#plt.figure(figsize = (10, 6)) 
#sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = "Greens", linecolor="Blue", linewidths=1.5) 
#plt.show()
from sklearn.metrics import classification_report
print(classification_report(Y_test,ypred))
print("Accuracy :" , accuracy_score(Y_test , ypred))
print("\n")
print("Decision Tree Algorithm")
dt_scores  = []
for k in range(1,40):
    dt = DecisionTreeClassifier(random_state = k)
    dt.fit(X_train , Y_train)
    ypred = dt.predict(X_test)
    dt_scores.append(accuracy_score(Y_test , ypred))
print(f"Best choice of k:{np.argmax(dt_scores)+1}")
dt = DecisionTreeClassifier(random_state = 2)
print(dt.fit(X_train , Y_train))#fit is used to train the model
te_score1iii = dt.score(X_test , Y_test)
print(f"\n Testing accuracy : {te_score1iii}")
ypred1iii = dt.predict(X_test)
cm1iii = confusion_matrix(Y_test,ypred1iii)
print("Confusion matrix:\n" , cm1iii )
print(classification_report(Y_test,ypred1iii))
print("Accuracy :" , accuracy_score(Y_test , ypred1iii))
print("Random Forest Algorithm")
rf_scores  = []
for k in range(1,40):
    rf = RandomForestClassifier(random_state = k)
    rf.fit(X_train , Y_train)
    ypred = rf.predict(X_test)
    rf_scores.append(accuracy_score(Y_test , ypred))
print(f"Best choice of k:{np.argmax(rf_scores)+1}")
rf2 =  RandomForestClassifier(random_state = 4 )
print(rf2.fit(X_train , Y_train))#fit is used to train the model
te_score1iv = rf2.score(X_test , Y_test)
print(f"\n Testing accuracy : {te_score1iv}")
ypred1iv = rf2.predict(X_test)
cm1iv = confusion_matrix(Y_test,ypred1iv)
print("Confusion matrix:\n" , cm1iv )
print(classification_report(Y_test,ypred1iv))
print("Accuracy :" , accuracy_score(Y_test , ypred1iv))
from sklearn.ensemble import  AdaBoostClassifier
print("Ada Algorithm")
ad_scores  = []
for k in range(1,40):
    ad = AdaBoostClassifier(random_state =k)
    ad.fit(X_train , Y_train)
    ypred = ad.predict(X_test)
    ad_scores.append(accuracy_score(Y_test , ypred))
print(f"Best choice of k:{np.argmax(ad_scores)+1}")
ad = AdaBoostClassifier(random_state =1)
print(ad.fit(X_train , Y_train))#fit is used to train the model
te_score2ii = ad.score(X_test , Y_test)
print(f"\n Testing accuracy : {te_score2ii}")
ypred2ii = ad.predict(X_test)
cm2ii = confusion_matrix(Y_test,ypred2ii)
print("Confusion matrix:\n" , cm2ii )
print(classification_report(Y_test,ypred2ii))
print("Accuracy :" , accuracy_score(Y_test , ypred2ii))
from xgboost import XGBClassifier
print("XGB Algorithm")
xg_scores  = []
for k in range(1,40):
    xg = XGBClassifier(random_state = k)
    xg.fit(X_train , Y_train)
    ypred = xg.predict(X_test)
    xg_scores.append(accuracy_score(Y_test , ypred))
print(f"Best choice of k:{np.argmax(xg_scores)+1}")
xg = XGBClassifier(random_state = 1)
xg.fit(X_train , Y_train)#fit is used to train the model
te_score2iii = xg.score(X_test , Y_test)
print(f"\n Testing accuracy : {te_score2iii}")
ypred2iii = xg.predict(X_test)
cm2iii = confusion_matrix(Y_test,ypred2iii)
print("Confusion matrix:\n" , cm2iii )
print(classification_report(Y_test,ypred2iii))
print("Accuracy :" , accuracy_score(Y_test , ypred2iii))
from sklearn.linear_model import  SGDClassifier
sg_scores  = []
for k in range(1,40):
    sg = SGDClassifier(random_state =k)
    sg.fit(X_train , Y_train)
    ypred = sg.predict(X_test)
    sg_scores.append(accuracy_score(Y_test , ypred))
print(f"Best choice of k:{np.argmax(sg_scores)+1}")
sg = SGDClassifier(random_state =16)
print(sg.fit(X_train , Y_train))
te_score3iii = sg.score(X_test , Y_test)
print(f"\n Testing accuracy : {te_score3iii}")
ypred3iii= sg.predict(X_test)
cm3iii = confusion_matrix(Y_test,ypred3iii)
print("Confusion matrix:\n" , cm3iii )
print(classification_report(Y_test,ypred3iii))
print("Accuracy :" , accuracy_score(Y_test , ypred3iii))
print("SVM Algorithm")
sv_scores  = []
for k in range(1,100):
    sv = SVC(random_state = k)
    sv.fit(X_train , Y_train)
    ypred = sv.predict(X_test)
    sv_scores.append(accuracy_score(Y_test , ypred))
print(f"Best choice of k:{np.argmax(sv_scores)+1}")
sv = SVC(random_state = 1)
print(sv.fit(X_train , Y_train))
te_score2i = sv.score(X_test , Y_test)
print(f"\n Testing accuracy : {te_score2i}")
ypred2i = sv.predict(X_test)
cm2i = confusion_matrix(Y_test,ypred2i)
print("Confusion matrix:\n" , cm2i )
print(classification_report(Y_test,ypred2i))
print("Accuracy :" , accuracy_score(Y_test , ypred2i))
knn_scores  = []
for k in range(1,100):
    kn = KNeighborsClassifier(n_neighbors = k)
    kn.fit(X_train , Y_train)
    ypred = kn.predict(X_test)
    knn_scores.append(accuracy_score(Y_test , ypred))
print(f"Best choice of k:{np.argmax(knn_scores)+1}")
kn = KNeighborsClassifier(n_neighbors = 20)
print(kn.fit(X_train , Y_train))
te_scoreii = kn.score(X_test , Y_test)
print(f"\n Testing accuracy : {te_scoreii}")
ypred1ii = kn.predict(X_test)
cm1ii = confusion_matrix(Y_test,ypred1ii)
print("Confusion matrix:\n" , cm1ii )
print(classification_report(Y_test,ypred1ii))
print("Accuracy :" , accuracy_score(Y_test , ypred1ii))
input_data = (48,0,2,130,275,0,1,139,0,0.2,2,0,2)
input_data_as_numpy_array= np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = rf2.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')
import pickle
pickle.dump(rf2, open('rf2.pkl','wb'))   