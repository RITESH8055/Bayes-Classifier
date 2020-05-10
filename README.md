# Bayes-Classifier
Using different covariance matrices
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


df= pd.read_csv(r'C:/Users/Asus/Documents/PRML/Dataset_1_Team_39.csv')
x=df[['# x_1','x_2','Class_label']]
y=df[['Class_label']]
z=df[['# x_1','x_2']]
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 4)
x[:2]
L  =np.array([[0,2,1],[2,0,3],[1,3,0]])

indices_2 = yTrain[yTrain == 2]
indices_1 = yTrain[yTrain == 1]
indices_0 = yTrain[yTrain == 0]


df0 = xTrain[xTrain['Class_label'] == 0]
plt.scatter(df0['# x_1'], df0['x_2'], color='r')
df1 = xTrain[xTrain['Class_label'] == 1]
plt.scatter(df1['# x_1'], df1['x_2'], color='b')
df2 = xTrain[xTrain['Class_label'] == 2]
plt.scatter(df2['# x_1'], df2['x_2'], color='green')


size=len(df)
P0=len(df0)/size
P1=len(df1)/size
P2=len(df2)/size


mu_0 = np.array(np.mean(df0))[:2]
mu_1 = np.array(np.mean(df1))[:2]
mu_2 = np.array(np.mean(df2))[:2]
mu_0



cov_0 = np.cov(df0['# x_1'],df0['x_2'])
cov_1 = np.cov(df1['# x_1'],df1['x_2'])
cov_2 = np.cov(df2['# x_1'],df2['x_2'])


cov10=np.array([[1,0],[0,1]])
cov11=np.array([[1,0],[0,1]])
cov12=np.array([[1,0],[0,1]])


cov20= P0*np.array([[cov_0[0,0],0],[0,cov_0[1,1]]]) + P1*np.array([[cov_1[0,0],0],[0,cov_1[1,1]]]) + P2*np.array([[cov_2[0,0],0],[0,cov_2[1,1]]])
cov21= cov20
cov22= cov20


cov30 = np.array([[cov_0[0,0],0],[0,cov_0[1,1]]])  
cov31= np.array([[cov_1[0,0],0],[0,cov_1[1,1]]])                                 
cov32= np.array([[cov_2[0,0],0],[0,cov_2[1,1]]])



cov40= P0 * cov_0 + P1* cov_1 + P2* cov_2
cov41= cov40
cov42= cov40



cov50=cov_0
cov51=cov_1                                
cov52=cov_2

def bivariate_normal(X,mu,cov):
    p = np.exp(((np.dot(np.dot(np.transpose(X-mu),np.linalg.inv(cov)),(X-mu))*(-0.5))))/(np.sqrt(np.pi *2*np.linalg.det(cov)))
    return(p)
 b = bivariate_normal(,mu_0,cov10)

def Bayes_Classifier(d,mu0,mu1,mu2,cov0,cov1,cov2):
    f0 = bivariate_normal(d,mu0,cov0)
    f1 = bivariate_normal(d,mu1,cov1)
    f2 = bivariate_normal(d,mu2,cov2)
    q0  = P0*f0/(P0*f0 + P1*f1+ P2*f2)
    q1  = P1*f1/(P0*f0 + P1*f1+ P2*f2)
    q2  = P2*f2/(P0*f0 + P1*f1+ P2*f2)
    q= np.array([q0,q1,q2])
    q = q[:, 0, 0]
    R0 = np.asscalar(np.dot(L[0,:],q))
    R1 = np.asscalar(np.dot(L[1,:],q))
    R2 =np.asscalar(np.dot(L[2,:],q))
    if (R0 < R1) & (R0 < R2):
        Class = 0
    elif (R1 <= R0) & (R1 < R2):
        Class = 1
    else:
        Class = 2
        
    return(Class)

vec_bayes_classifier = np.vectorize(Bayes_Classifier)

X_train = np.array(xTrain[["# x_1", "x_2"]])

Y_train = np.array(yTrain["Class_label"])
b = bivariate_normal(X_train[0], mu_2, cov10)
print(b)

Y_pred = np.zeros(len(X_train))

for i in range(len(Y_pred)):
    Y_pred[i] = Bayes_Classifier(X_train[i], mu_0, mu_1, mu_2, np.matrix(cov10), np.matrix(cov11), np.matrix(cov12))


indices_0 = Y_pred == 0
indices_1 = Y_pred == 1
indices_2 = Y_pred == 2

plt.scatter(X_train[indices_0, 0], X_train[indices_0, 1], c="r")
plt.scatter(X_train[indices_1, 0], X_train[indices_1, 1], c="b")
plt.scatter(X_train[indices_2, 0], X_train[indices_2, 1], c="g")
                                                                                         
                                                                                         
C = confusion_matrix(pred_test_diff_bayes,yTest)
Y_pred = np.zeros(len(X_train))
for i in range(len(Y_pred)):
    Y_pred[i] = Bayes_Classifier(X_train[i], mu_0, mu_1, mu_2, np.matrix(cov50), np.matrix(cov51), np.matrix(cov52))
indices_0 = Y_pred == 0
indices_1 = Y_pred == 1
indices_2 = Y_pred == 2

plt.scatter(X_train[indices_0, 0], X_train[indices_0, 1], c="r")
plt.scatter(X_train[indices_1, 0], X_train[indices_1, 1], c="b")
plt.scatter(X_train[indices_2, 0], X_train[indices_2, 1], c="g")  
def Bayes_Classifier(d,mu0,mu1,mu2,cov0,cov1,cov2):
     f0 = np.random.multivariate_normal(d,mu0,cov0)
     f1 = np.random.multivariate_normal(d,mu1,cov1)
     f2 = np.random.multivariate_normal(d,mu2,cov2)
    q0  = P0*f0/(P0*f0 + P1*f1+ P2*f2)
    q1  = P1*f1/(P0*f0 + P1*f1+ P2*f2)
    q2  = P2*f2/(P0*f0 + P1*f1+ P2*f2)
    q= np.array(q0,q1,q2)
    R0 = np.asscalar(np.dot(L[0,:],q))
    R1 = np.asscalar(np.dot(L[1,:],q))
    R2 =np.asscalar(np.dot(L[2,:],q))
    if (R0 < R1) & (R0 < R2):
        Class = 0
    elif (R1 <= R0) & (R1 < R2):
        Class = 1
    else:
        Class = 2
        
    return(Class)
    
def predictor(X,m0,m1,m2,cov0,cov1,cov2,t_label):
    X=np.matrix(X)
    pred = np.zeros(np.shape(X)[0])
    for i in range(1,np.shape(X)[0]+1):
        #pred[i] += Bayes_Classifier(X[(i-1):i],m0,m1,m2,cov0,cov1,cov2)
    r = 0
    w = 0
    for i in range(0,np.shape(pred)[0]):
        if pred[i]==t_label[i]:
            r += 1
        else:
            w += 1
            
    Train_accuracy = r/(np.shape(pred)[0])  
    print(Train_accuracy)
    return(pred,Train_accuracy)
