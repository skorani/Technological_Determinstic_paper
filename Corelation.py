#date
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np



read_data = pd.read_csv("First_set.csv",  sep=',' , names=['idx','userId','time','topic1','topic2','topic3','topic4','topic5','topic6','topic7','topic8','topic9','topic10','topic11','topic12','topic13','topic14','topic15','topic16','topic17','topic18','topic19','topic20','topic21','topic22','topic23','topic24','topic25','topic26','topic27','topic28','topic29','topic30','topic31','topic32','topic33','topic34','topic35'],index_col=0)
print(read_data.head())
df = pd.concat([pd.Series(1, index=read_data.index, name='00'), read_data], axis=1)
df.head()
data = df.groupby(['topic1','topic2','topic3','topic4','topic5','topic6','topic7','topic8','topic9','topic10','topic11','topic12','topic13','topic14','topic15','topic16','topic17','topic18','topic19','topic20','topic21','topic22','topic23','topic24','topic25','topic26','topic27','topic28','topic29','topic30','topic31','topic32','topic33','topic34','topic35'])
X = df[['topic5','topic6']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y = df['topic1']
for i in range(1, len(X.columns)):
    X[i-1] = X[i-1]/np.max(X[i-1])
X.head()
theta = np.array([0]*len(X.columns))
m = len(df)
def hypothesis(theta, X):
    return theta*X
def computeCost(X, y, theta):
    y1 = hypothesis(theta, X)
    y1=np.sum(y1, axis=1)
    return sum(np.sqrt((y1-y)**2))/(2*47)
def gradientDescent(X, y, theta, alpha, i):
    J = []  #cost function in each iterations
    k = 0
    while k < i:        
        y1 = hypothesis(theta, X)
        y1 = np.sum(y1, axis=1)
        for c in range(0, len(X.columns)):
            theta[c] = theta[c] - alpha*(sum((y1-y)*X.iloc[:,c])/len(X))
        j = computeCost(X, y, theta)
        J.append(j)
        k += 1
    return J, j, theta
J, j, theta = gradientDescent(X, y, theta, 0.05, 10000)
y_hat = hypothesis(theta, X)
y_hat = np.sum(y_hat, axis=1)
plt.figure()
plt.scatter(x=list(range(0, 47)),y= y, color='blue')         
plt.scatter(x=list(range(0, 47)), y=y_hat, color='black')
plt.show()
plt.figure()
plt.scatter(x=list(range(0, 10000)), y=J)
plt.show()