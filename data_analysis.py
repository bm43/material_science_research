import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from kneed import KneeLocator
directory='C:/Users/SamSung/Desktop/OneDrive_2020-06-17/6_PDMS_Donuts/cleanerdata'
filepath='C:/Users/SamSung/Desktop/OneDrive_2020-06-17/6_PDMS_Donuts/cleanerdata/2.xlsx'
filepath2='C:/Users/SamSung/Desktop/OneDrive_2020-06-17/6_PDMS_Donuts/cleanerdata/2.xlsx'
def reg(filepath):
    #threshold=7
    data=pd.read_excel(filepath)

    #$disp_limit=-1.56

    data=data.drop(['Unnamed: 0'],axis=1)

    # .gt() tells true/false when strictly greater than ...

    #MA=dataf.rolling(window=10).mean()
    #plt.plot(dataf.Disp,MA,color='red')

    #X=preprocessing.scale(X)
    X=data.iloc[:,0].values.reshape(-1,1).ravel()
    Y=data.iloc[:,1].values.reshape(-1,1).ravel()
    #Y=preprocessing.scale(Y)
    #Y2=data2.iloc[:,1].values.reshape(-1,1)
    kneedle=KneeLocator(X,Y,S=1,curve='concave',direction='increasing',interp_method="interp1d")
    threshold=round(kneedle.knee,3)
    data=data[data.Disp<threshold]
    #print(data['Load'].idxmax())
    data.plot('Disp','Load')
    #print(data.index)
    #print(data['Load'].idxmin())
    #print(data['Load'].min())
    data=data[data.index < data['Load'].idxmin()]
    #-> removes higher curve
    #print(data['Load'].idxmax())
    data.plot(x='Disp',y='Load')

    #svr=SVR(kernel='poly',degree=12,C=0.1,epsilon=0.01)
    # ->this gives perfect fit to 3rd xlsx file
    #sklearn ver 0.20.4
    X=data.iloc[:,0].values.reshape(-1,1)
    Y=data.iloc[:,1].values.reshape(-1,1)
    """
    xx=np.linspace(-1,0,num=68)
    plt.plot(xx,Y)
    """
    #############lin reg###################
    lr=LinearRegression()
    model=lr.fit(X,Y)
    prediction=model.predict(X)

    #plt.plot(X,prediction,color='red')

    #######################################
    #plt.show()

    return 0

reg(filepath)


"""
python read files in following order: 10,11,12,1,2,3,4....
so changed it into 1,2,3,...,11,12

"""
