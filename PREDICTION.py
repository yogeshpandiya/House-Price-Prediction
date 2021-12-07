import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

df1=pd.read_csv("C:/Users/yogesh/Desktop/Machine Learning/new.csv")
print(df1)



reg= linear_model.LinearRegression()
reg.fit(df1[['Area']].values,df1.Price)

test=np.array([210,320,390])
#Test=test[np.newaxis, 4:5]
#print(reg.predict(Test))

a=[]
for i in range(len(test)):
    #print(i)
    Test=test[np.newaxis, i:i+1]
    b=reg.predict(Test)
    #print(b)
    #c=b.tolist()
    #print(type(c))
    a.append(b[0])

#print(a)
df = pd.DataFrame()
df['Area'] = test[0::]
df['Price'] = a[0::]
df.to_csv('C:/Users/yogesh/Desktop/Machine Learning/new123.csv', index = False)

plt.scatter(df.Area, df.Price)
plt.plot(df1.Area, df1.Price)
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()
