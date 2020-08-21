#!/usr/bin/env python
# coding: utf-8

# In[ ]:


df_sort.drop(['hospitalizedCurrently', 'hospitalizedCumulative', 'recovered', 'checkTimeEt', 'hospitalized', 
         'positiveCasesViral', 'pending', 'inIcuCurrently', 'inIcuCumulative', 'onVentilatorCurrently', 'onVentilatorCumulative', 'dataQualityGrade',
        'lastUpdateEt', 'dateModified', 'dateChecked', 'totalTestsViral', 'positiveTestsViral', 'negativeTestsViral', 
        'deathConfirmed', 'deathProbable', 'fips', 'positiveIncrease', 'negativeIncrease', 'totalTestResults', 
        'totalTestResultsIncrease', 'posNeg', 'deathIncrease', 'hospitalizedIncrease', 'hash', 'commercialScore',
        'negativeRegularScore', 'negativeScore', 'positiveScore', 'score', 'grade'], axis=1)


# In[42]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression


# In[33]:


def toDate(iDate):
    sDate = str(iDate)
    years = int(sDate[0:4])
    months = int(sDate[4:6])
    days = int(sDate[6:8])
    date = datetime.datetime(years, months, days)
    return date


# In[38]:


cd = pd.read_csv('usa_states_covid19_daily.csv', error_bad_lines=False)
sd = pd.read_csv('state_populations.csv', error_bad_lines=False)
cd['date'].apply(toDate)
df_sort = cd.sort_index(ascending=True, axis=0)
df = df_sort.groupby('state')


# In[35]:


sd.head(5)


# In[27]:


df.get_group('AK')['positiveIncrease'].agg(np.mean)


# In[28]:


def rateOfIncrease(state):
    mean = df.get_group(state)['positiveIncrease'].agg(np.mean)
    return mean


# In[29]:


rateOfIncrease('AK')


# In[36]:


sd.head()


# In[41]:


for state in sd['State']:
    print(state, rateOfIncrease(state)/Population)


# In[81]:


def predict(state):
    Y = df.get_group(state)['positiveIncrease'].values.reshape(-1, 1)
    X = df.get_group(state)['date'].values.reshape(-1, 1)
    linear_regressor = LinearRegression()
    linear_regressor.fit(X, Y)
    Y_pred = linear_regressor.predict(X)
    print(linear_regressor.score(X,Y))
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.show()


# In[89]:


predict('AK')


# In[ ]:




