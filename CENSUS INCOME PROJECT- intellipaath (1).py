#!/usr/bin/env python
# coding: utf-8

# # Problem Statement:
#  In this project, initially you need to preprocess the data and then develop an
#  understanding of the different features of the data by performing exploratory
#  analysis and creating visualizations. Further, after having sufficient knowledge
#  about the attributes, you will perform a predictive task of classification to predict
#  whether an individual makes over 50,000 a year or less by using different
#  machine learning algorithms.

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt    


# In[2]:


df = pd.read_csv(r"C:\Users\DELL\Desktop\census-income (7).csv")


# In[3]:


df


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df.describe()


# In[7]:


# Rename the last column to 'income'
df.rename(columns={df.columns[-1]: 'income'}, inplace=True)


# In[8]:


print(df['income'].unique())


# In[9]:


print(df['income'].value_counts())


# In[10]:


# Define the mapping for the income column
income_mapping = {
    '>50K': 1,
    '<=50K': 0
}


# In[11]:


print(df['income'].value_counts())


# In[12]:


df.info()


# In[13]:


df.head()


# In[14]:


print(df.columns)


# In[15]:


# Renaming columns if needed
df.columns = [col.strip() for col in df.columns]  # Remove any leading/trailing spaces


# In[16]:


df = pd.get_dummies(df, columns=['workclass', 'education', 'marital-status', 
                                      'occupation', 'relationship', 'race', 
                                      'sex', 'native-country'], drop_first=True)


# In[17]:


X = df.drop('income', axis=1)  # Features
y = df['income']                # Target variable


# In[18]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 2. Exploratory Data Analysis (EDA)

# In[19]:


# Summarize numerical features
print(df.describe())


# Visualizations

# In[20]:


sns.histplot(df['age'], bins=30)
plt.title('Age Distribution')
plt.show()


# In[21]:


sns.countplot(x='income', data=df)
plt.title('Income Distribution')
plt.show()


# The '≤50k' income category shows a higher concentration of individuals

# In[39]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score


# In[40]:


df


# In[41]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[42]:


log_model =LogisticRegression()


# In[43]:


log_model.fit(X_train,y_train)


# In[44]:


y_pred = log_model.predict(X_test)


# In[45]:


conf_matrix = confusion_matrix(y_pred,y_test)
acc_score = accuracy_score(y_test,y_pred)


# In[46]:


print("Confusion Matrix:")
print(conf_matrix)


# In[47]:


print(f"Accuracy Score: {acc_score:.2f}")


# In[48]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[49]:


rt_model =RandomForestClassifier()


# In[50]:


rt_model.fit(X_train,y_train)


# In[51]:


y_pred = rt_model.predict(X_test)


# In[52]:


y_pred


# In[53]:


conf_matrix = confusion_matrix(y_pred,y_test)
acc_score = accuracy_score(y_test,y_pred)


# In[54]:


print("Confusion Matrix:")
print(conf_matrix)


# In[55]:


print(f"Accuracy Score: {acc_score:.2f}")

