#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


housing=pd.read_csv("data.csv")


# In[3]:


housing.info()


# In[4]:


housing.head()


# In[5]:


housing.describe()


# In[6]:


import numpy as np


# In[7]:


def split_train_test(data,test_ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]
    


# In[8]:


train_set,test_set=split_train_test(housing,0.2)


# In[9]:


print(len(train_set),len(test_set))


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)


# In[12]:


print("rows in train:",len(train_set))
print("rows in test:",len(test_set))


# In[13]:


from sklearn.model_selection import StratifiedShuffleSplit


# In[14]:


split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)


# In[15]:


for train_index,test_index in split.split(housing,housing["CHAS"]):
    strat_train_set=housing.iloc[train_index]
    strat_test_set=housing.iloc[test_index]


# In[16]:


print(len(strat_train_set))


# In[17]:


corr_matrix=housing.corr()


# In[18]:


print(corr_matrix)


# In[19]:


corr_matrix["MEDV"].sort_values(ascending=False)


# In[20]:


from pandas.plotting import scatter_matrix


# In[21]:


attributes=["MEDV","RM","ZN","LSTAT"]
scatter_matrix(housing[attributes],figsize=(10,8))


# In[22]:


housing.plot(kind="scatter",x="RM",y="MEDV",alpha=(0.8))


# In[23]:


housing=strat_train_set.drop("MEDV",axis=1)
housing_labels=strat_train_set["MEDV"].copy()


# In[24]:


from sklearn.pipeline import Pipeline


# In[25]:


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy="median")
imputer.fit(housing)
x=imputer.transform(housing)


# In[26]:


from sklearn.preprocessing import StandardScaler
my_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('std_scalar',StandardScaler())
])


# In[27]:


housing_tr=my_pipeline.fit_transform(housing)


# In[28]:


print(len(housing_tr))


# In[29]:


print(len(train_set))


# In[30]:


from sklearn.ensemble import RandomForestRegressor


# In[31]:


model=RandomForestRegressor()


# In[32]:


model.fit(housing_tr,housing_labels)


# In[33]:


some_data=housing.iloc[:5]
some_labels=housing.iloc[:5]


# In[34]:


prepared_data=my_pipeline.transform(some_data)


# In[35]:


model.predict(prepared_data)


# In[36]:


list(some_labels
    )


# In[37]:


from sklearn.metrics import mean_squared_error


# In[38]:


housing_predictions=model.predict(housing_tr)


# In[39]:


mse=mean_squared_error(housing_labels,housing_predictions)


# In[40]:


rmse=np.sqrt(mse)


# In[41]:


print(rmse)


# In[42]:


from sklearn.model_selection import cross_val_score


# In[43]:


scores=cross_val_score(model,housing_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)


# In[44]:


rmse_scores=np.sqrt(-scores)


# In[45]:


def print_scores(scores):
    print("scores: ", scores)
    print("mean: ",scores.mean())
    print("deviation: ", scores.std())


# In[46]:


print_scores(rmse_scores)


# In[47]:


from joblib import dump,load


# In[48]:


dump(model,"modeltest.joblib")


# In[49]:


x_test=strat_test_set.drop("MEDV",axis=1)


# In[50]:


y_test=strat_test_set["MEDV"].copy()


# In[51]:


x_test_prepared=my_pipeline.transform(x_test)
final_predictions=model.predict(x_test_prepared)


# In[52]:


final_mse=mean_squared_error(y_test,final_predictions)
final_rmse=np.sqrt(final_mse)


# In[54]:


print(final_rmse)


# In[55]:


print(prepared_data)


# In[56]:


from joblib import dump,load


# In[57]:


dump(model,"house.joblib")


# In[ ]:




