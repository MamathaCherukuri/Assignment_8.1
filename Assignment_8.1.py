#!/usr/bin/env python
# coding: utf-8

# 1.Read the following data set

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


Adult=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')


# In[11]:


Adult


# 2.Rename the columns as per the description from this file:

# In[5]:


Adult.columns=['age','workclass','fnlwgt','education','education_num',
               'marital_status',
               'occupation','relationship','race','sex','capital_gain',
               'capital_loss','hours_per_week','native_country','Amount']


# In[6]:


Adult.head()


# In[7]:


from pandas import DataFrame, Series
import sqlite3 
from pandasql import sqldf as sql
import sys


# Create a sql db from adult dataset and name it sqladb

# In[8]:


conn = sqlite3.connect('TestDB.db')


# In[ ]:


3.fCreate a sql db from adult dataset and name it sqladb


# In[9]:


c = conn.cursor()
 


# In[12]:


Adult.to_sql("Adult.csv", conn, if_exists="replace")


# In[13]:


conn.execute(
    """
    create table my_table as 
    select * from Adult
    """)


# In[18]:


df = pd.read_sql_query("select * from Adult limit 10;", conn)
df


# Show me the average hours per week of all men who are working in private sector

# In[27]:


df1 = pd.read_sql_query('SELECT hours_per_week,sex, avg(hours_per_week) as `Avg_hours`'
                       'FROM Adult '
                       'GROUP BY sex ', conn)


# In[28]:


df1


# Show me the frequency table for education, occupation and relationship, separately

# In[44]:


sql1= """SELECT education,COUNT(*) as cnt
         FROM Adult
         GROUP BY education;"""
df2= pd.read_sql_query(sql1, conn)


# In[45]:


df2


#  occupation:Frequency Table

# In[46]:


sql2= """SELECT occupation,COUNT(*) as cnt
         FROM Adult
         GROUP BY occupation;"""
df3= pd.read_sql_query(sql2, conn)
df3


# relationship:Frequency Table

# In[47]:


sql3= """SELECT relationship,COUNT(*) as cnt
         FROM Adult
         GROUP BY relationship;"""
df4= pd.read_sql_query(sql3, conn)
df4


# 4. Are there any people who are married, working in private sector and having a masters degree

# In[49]:


sql4= """SELECT marital_status,education,workclass,COUNT(*) as cnt
         FROM Adult
         Where education="Masters" & workclass="Private";"""
df5= pd.read_sql_query(sql4, conn)
df5


# 5. What is the average, minimum and maximum age group for people working in different sectors

# In[51]:


sql5= """SELECT occupation,min(age) as min_age,max(age) as max_age, avg(age) as avg_age
         FROM Adult
         GROUP BY occupation;"""
df6= pd.read_sql_query(sql5, conn)
df6


# 6. Calculate age distribution by country

# In[53]:


sql6= """SELECT age,native_country,count(*) as cnt
         FROM Adult
         GROUP BY native_country;"""
df7= pd.read_sql_query(sql6, conn)
df7


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




