# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 14:08:44 2022

@author: Rushikesh
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('D:/SEML/Dataset/IRIS.csv')
df1 = df

df['species'].unique()
df1['species'] = df1['species'].replace({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})

# Logistic Regression
"---------------------Logistic Regression------------------------------"
x = df.iloc[:,[0.1,2,3]]
x1 = x.values
y = df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_tr,x_te,y_tr,y_te = train_test_split(x1,y,test_size=0.25,random_state=0)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver = 'liblinear')

logreg.fit(x_tr,y_tr)

y_pr = logreg.predict(x_te)
from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(y_te,y_pr))
print(accuracy_score(y_te,y_pr))

# Support Vector Machines :: SVM
"---------------------Support Vector Machines------------------------------"
from sklearn.svm import SVC
clf = SVC(kernel = 'linear').fit(x_tr,y_tr)
clf.predict(x_tr)
y_pred = clf.predict(x_te)
cm = confusion_matrix(y_te, y_pred)
print(accuracy_score(y_te,y_pred))
import seaborn as sns
cm_df = pd.DataFrame(cm,index = ['SETOSA','VERSICOLR','VIRGINICA'],columns = ['SETOSA','VERSICOLR','VIRGINICA'])
plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()

# KMeans Algorithm
"---------------------KMeans Algorithm------------------------------"

from sklearn.cluster import KMeans
wcss =[]
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Within Clusters of Sum of Squares')

kmeans = KMeans(n_clusters=3,init ='k-means++',random_state=0)
X=x.values
y_kmeans = kmeans.fit_predict(X)
centroid = kmeans.cluster_centers_
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,marker='^',color ='purple',label='IRIS-Setosa')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,marker='d',color ='grey',label='IRIS-Versicolor')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,marker='*',color ='olive',label='IRIS-Virginica')
plt.scatter(centroid[:,0],centroid[:,1],s=300,marker='o',color='red',alpha=0.5,label='Centroids')
plt.legend(loc=4)
plt.title('K-Means Cluster of IRIS')

# Performing Sillhoute Scores Analysis on IRIS
from sklearn.metrics import silhouette_score
sil_score = []

for i in range(3,11): # Started from range of 3 because there are min 3 categori
     kmeans = KMeans(n_clusters = i,init='k-means++',random_state=0)
     kmeans.fit(x)
     cluster_labels = kmeans.labels_
     sil_score.append(silhouette_score(x,cluster_labels))
     print('%f : %f'%(i,silhouette_score(x,cluster_labels)))
     
# Hierarchial Clustering on IRIS

'****************Hierarchial Clustering************************'

from scipy.cluster import hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Visualizing Dendrogram

dendrogram_p = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendrogram for Iris')

# It can be readily visible that 3 Clusters can be easily formed
# This results in morever equalised sized clusters

hi = AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
y_hi = hi.fit_predict(X)

plt.scatter(X[y_hi == 0, 0], X[y_hi == 0, 1], s = 100,marker='^',color ='purple',label='IRIS-Setosa')
plt.scatter(X[y_hi == 1, 0], X[y_hi == 1, 1], s = 100, marker='d',color ='grey',label='IRIS-Versicolor')
plt.scatter(X[y_hi == 2, 0], X[y_hi == 2, 1], s = 100, marker='*',color ='olive',label='IRIS-Virginica')
plt.legend()


fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,6))
ax[0].scatter(X[:,0],X[:,1],c=y_kmeans,cmap='viridis',)
ax[1].scatter(X[:,0],X[:,1],c=y_hi,cmap='plasma',title='Hierarchial')

" **************** All Classifier in One Go ***************** "

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()
voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],voting='hard')
voting_clf.fit(x_tr, y_tr)

from sklearn.metrics import accuracy_score
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(x_tr,y_tr)
    y_pred = clf.predict(x_te)
    print(clf.__class__.__name__, accuracy_score(y_te, y_pred))

"***************Bagging and Pasting*******************"

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,max_samples=100, bootstrap=True, n_jobs=-1e)
bag_clf.fit(x_tr, y_tr)
y_pred = bag_clf.predict(x_te)
accuracy_score(y_te, y_pred)

'Bagging with oob scores'
'Out of Bag Scores estimate what percent accuracy the model can achieve with test set'
''
bag_clf = BaggingClassifier(DecisionTreeClassifier(), bootstrap=True, n_jobs=-1,oob_score=True)
bag_clf.fit(x_tr, y_tr)
y_pred = bag_clf.predict(x_te)
accuracy_score(y_te, y_pred)
bag_clf.oob_score_
print(accuracy_score(y_te, y_pred))
bag_clf.oob_decision_function_ # the class-wise probability for each instance
" ''''''Naive - Bayes Classification'''''''"

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_tr,y_tr)
# making predictions on the testing set
y_pred = gnb.predict(x_te)
metrics.confusion_matrix(y_te, y_pred)
# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_te, y_pred)*100)
