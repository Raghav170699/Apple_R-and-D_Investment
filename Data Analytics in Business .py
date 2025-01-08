#!/usr/bin/env python
# coding: utf-8

# In[133]:


import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np


# In[134]:


#Apple Inc
df = pd.read_csv('Sample 1_CSV_S&P1500_Raw Dataset_Data Analytics in Business Assignment_2021.csv')
df = df.rename(columns={'R&D Expense':'Rnd','Total Assets':'Tot', 'Number of Employees':'Emp','Social Disclosure Score':'SDS', 'Return on Assets':'ROA'})
df_B = df.loc[df['Name'] == 'Apple Inc']
dfb = df_B[['Name','Rnd','Emp','Tot','SIC Code','Year']]
dfb.describe()


# In[135]:


#Data of Competitors
dfcomp1 = df[(df['SIC Code'] >= 3570) & (df['SIC Code'] <= 3579)]
dfcomp2 = dfcomp1[['Name','Rnd','Emp','Tot','SIC Code','Year']]
dfc=dfcomp2.dropna()
dfc


# In[136]:


dfc.describe()


# In[ ]:


#Scatterplot 
scatter, ax = plt.subplots()
ax = sns.regplot(x = 'Rnd', y = 'Tot', data = dfc)
plt.show()


# In[ ]:


import seaborn as sns

# Create a horizontal bar plot
dfc.plot.barh(x='Name', y='Emp', legend=False)

plt.rcParams["figure.figsize"] = (50,0)


# Show the plot
plt.show()


# In[ ]:





# In[99]:


sns.barplot(x='Tot', y='Name', data=dfc, orient='h', color='black')
plt.show


# In[54]:


#Regression Analysis
y = dfc.Tot
X = dfc[["Rnd","Emp"]].assign(const=1)
model = sm.OLS(y,X)
results = sm.OLS(y,X).fit()
print(results.summary())


# In[56]:


#Predictive Analysis 6423, 136829

data = {'Rnd': 15633 , 'Emp': 137333},
df = pd.DataFrame(data).assign(const=1)

results.predict(df)


# In[113]:


plt.figure(figsize=(8,5))
correlation = dfc.corr().round(4)
sns.heatmap(data=correlation, annot=True)
plt.show


# In[ ]:





# In[2]:


import numpy as np
import pandas as pd

df = pd.read_csv('Sample 1_CSV_S&P1500_Raw Dataset_Data Analytics in Business Assignment_2021.csv')

#create a dataset with only the companies that have sic codes between 3570 and 3579
df = df[(df['SIC Code'] >= 3570) & (df['SIC Code'] <= 3579)]

#show df
df

#create a dataset with unique names in df
df_unique = df.drop_duplicates(subset=['Name'])




#create a seaborn horizontal bar chart of the number of employees in each company in df_unique with the color of the bars being blue
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
ax = sns.barplot(x="Number of Employees", y="Name", data=df_unique, color='black', order=df_unique.sort_values('Number of Employees', ascending=False)['Name'])

#change the color of the company with Name = 'Apple Inc.' to red
ax.patches[2].set_color('red')


# In[3]:


ax2 = sns.barplot(x="R&D Expense", y="Name", data=df_unique, color='black', order=df_unique.sort_values('R&D Expense', ascending=False)['Name'])
ax2.patches[0].set_color('red')


# In[ ]:




