#!/usr/bin/env python
# coding: utf-8

# # Breakdown of this notebook:
# 1. **Loading the dataset:** Load the data and import the libraries. <br>
# 2. **Data Cleaning:** <br>
#  - Deleting redundant columns.
#  - Renaming the columns.
#  - Dropping duplicates.
#  - Cleaning individual columns.
#  - Remove the NaN values from the dataset
#  - #Some Transformations
# 
# 3. **Data Visualization:** Using plots to find relations between the features.
#  - Restaurants delivering Online or not
#  - Restaurants allowing table booking or not
#  - Table booking Rate vs Rate
#  - Best Location
#  - Relation between Location and Rating
#  - Restaurant Type
#  - Gaussian Rest type and Rating
#  - Types of Services
#  - Relation between Type and Rating
#  - Cost of Restuarant
#  - No. of restaurants in a Location
#  - Restaurant type
#  - Most famous restaurant chains in Bengaluru 
# 
# 
# 

# #### The basic idea is analyzing the <font color=blue>Buisness Problem of Zomato </font> to get a fair idea about the factors affecting the establishment of different types of restaurant at different places in Bengaluru, aggregate rating of each restaurant and many more.

# In[1]:


#Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score


# In[4]:


import os
os.getcwd()
os.chdir("C:\\Users\\Eshank\\Desktop\\")


# In[31]:


#reading the dataset
zomato_real=pd.read_csv("zomato.csv")
zomato_real.head() # prints the first 5 rows of a DataFrame


# In[6]:


zomato_real.info() # Looking at the information about the dataset, datatypes of the coresponding columns and missing values


# In[7]:


#Deleting Unnnecessary Columns
zomato=zomato_real.drop(['url','dish_liked','phone'],axis=1) #Dropping the column "dish_liked", "phone", "url" and saving the new dataset as "zomato"


# In[8]:


zomato.head() # looking at the dataset after transformation 


# In[9]:


#Removing the Duplicates
zomato.duplicated().sum()


# In[7]:


zomato.drop_duplicates(inplace=True)
zomato.head() # looking at the dataset after transformation


# In[10]:


#Remove the NaN values from the dataset
zomato.isnull().sum()


# In[11]:


zomato.dropna(how='any',inplace=True)
zomato.info() #.info() function is used to get a concise summary of the dataframe


# In[12]:


#Reading Column Names
zomato.columns


# In[13]:


#Changing the column names
zomato = zomato.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type',
                                  'listed_in(city)':'city'})
zomato.columns


# In[14]:


zomato["cost"].unique()


# In[15]:


#Some Transformations
zomato['cost'] = zomato['cost'].astype(str) #Changing the cost to string
zomato['cost'] = zomato['cost'].apply(lambda x: x.replace(',','.')) #Using lambda function to replace ',' from cost
zomato['cost'] = zomato['cost'].astype(float) # Changing the cost to Float
zomato.info() # looking at the dataset information after transformation


# In[16]:


#Reading uninque values from the Rate column
zomato['rate'].unique()


# In[ ]:





# In[17]:


zomato["cost"].unique()


# In[18]:


zomato["rate"].unique()


# In[19]:


#Removing '/5' from Rates
zomato.loc[zomato.rate =='NEW']


# In[20]:


zomato.loc[zomato.rate =='-']


# In[21]:


zomato = zomato.loc[zomato.rate !='-'].reset_index(drop=True)


# In[22]:


zomato.rate.unique()


# In[23]:


zomato.head(5)


# In[27]:


zomato2=zomato.copy()


# In[ ]:


zomato2["rate"]=zomato2["rate"].apply(lambda x: x.str[:3] )
zomato2["rate"]


# In[ ]:


remove_slash =zomato.rate.apply (lambda x: x.replace('/5', '') if type(x) == np.str else x)
zomato.rate = zomato.rate.apply(remove_slash).str.strip().astype('float')
zomato['rate'].head() # looking at the dataset after transformation


# In[33]:


# Adjust the column names
zomato.name = zomato.name.apply(lambda x:x.title())
zomato.online_order.replace(('Yes','No'),(True, False),inplace=True)
zomato.book_table.replace(('Yes','No'),(True, False),inplace=True)
zomato.head() # looking at the dataset after transformation


# In[15]:


zomato.cost.unique() # cheking the unique costs


# In[16]:


#Encode the input Variables
from sklearm.preprocssing import labelencoder()
def Encode(zomato):
    for column in zomato.columns[~zomato.columns.isin(['rate', 'cost', 'votes'])]:
        zomato[column] = zomato[column].factorize()[0]
    return zomato

zomato_en = Encode(zomato.copy())
zomato_en.head() # looking at the dataset after transformation


# In[17]:


#Get Correlation between different variables
corr = zomato_en.corr(method='kendall')
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True)
zomato_en.columns


# #### The highest correlation is between name and address which is 0.62 which is not of very much concern 

# # Regression Analysis

# ### Splitting the Dataset

# In[18]:


#Defining the independent variables and dependent variables
x = zomato_en.iloc[:,[2,3,5,6,7,8,9,11]]
y = zomato_en['rate']
#Getting Test and Training Set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=353)
x_train.head()


# In[19]:


y_train.head()


# In[20]:


zomato_en['menu_item'].unique() # seeing the unique values in 'menu_item'


# In[21]:


zomato_en['location'].unique() # seeing the unique values in 'location'


# In[22]:


zomato_en['cuisines'].unique() # seeing the unique values in 'cusines'


# In[23]:


zomato_en['rest_type'].unique() # seeing the unique values in 'rest_type'


# In[24]:


x.head()


# In[25]:


y.head()


# # Data Visualization

# #### Restaurants delivering Online or not

# In[33]:


#Restaurants delivering Online or not
sns.countplot(zomato['online_order'])
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('Restaurants delivering online or Not')


# #### Restaurants allowing table booking or not

# In[34]:


sns.countplot(zomato['book_table'])
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('Restaurants allowing table booking or not')


# #### Table booking Rate vs Normal Rate

# In[42]:



Y = pd.crosstab(zomato['rate'], zomato['book_table'])
Y


# In[47]:


plt.rcParams['figure.figsize'] = (17, 9)
Y.div(Y.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True,color=['red','green'])
plt.title('table booking vs Normal rate', fontweight = 30, fontsize = 20)
plt.legend(loc="upper right")
plt.show()


# #### Location

# In[48]:


sns.countplot(zomato['city'])
sns.countplot(zomato['city']).set_xticklabels(sns.countplot(zomato['city']).get_xticklabels(), rotation=90, ha="right")
fig = plt.gcf()
fig.set_size_inches(13,13)
plt.title('Location wise count for restaurants')


# #### Location and Rating

# In[50]:


loc_plt=pd.crosstab(zomato['rate'],zomato['city'])
loc_plt


# In[65]:


plt.rcParams['figure.figsize'] = (17, 9)
loc_plt.plot(kind='bar',stacked=True);
plt.title('Locationwise Rating',fontsize=15,fontweight='bold')
plt.ylabel('Location',fontsize=10,fontweight='bold')
plt.xlabel('Rating',fontsize=10,fontweight='bold')
plt.xticks(fontsize=10,fontweight='bold')
plt.yticks(fontsize=10,fontweight='bold');
plt.legend();


# #### Restaurant Type

# In[72]:


sns.countplot(zomato['rest_type'])
sns.countplot(zomato['rest_type']).set_xticklabels(sns.countplot(zomato['rest_type']).get_xticklabels(), rotation=90)
fig = plt.gcf()
fig.set_size_inches(16,8)
plt.title('Restuarant Type')


# #### Gaussian Rest type and Rating

# In[35]:


loc_plt=pd.crosstab(zomato['rate'],zomato['rest_type'])
loc_plt


# In[40]:


plt.rcParams['figure.figsize'] = (17, 9)
loc_plt.plot(kind='bar',stacked=True);
plt.title('Rest type - Rating',fontsize=15,fontweight='bold')
plt.ylabel('Rest type',fontsize=10,fontweight='bold')
plt.xlabel('Rating',fontsize=10,fontweight='bold')
plt.xticks(fontsize=10,fontweight='bold')
plt.yticks(fontsize=10,fontweight='bold');
plt.legend()


# #### Types of Services

# In[83]:


sns.countplot(zomato['type'])
sns.countplot(zomato['type']).set_xticklabels(sns.countplot(zomato['type']).get_xticklabels(), rotation=90)
fig = plt.gcf()
fig.set_size_inches(14,14)
plt.title('Type of Service')


# #### Type and Rating

# In[84]:


type_plt=pd.crosstab(zomato['rate'],zomato['type'])
type_plt


# In[85]:


type_plt.plot(kind='bar',stacked=True);
plt.title('Type - Rating',fontsize=15,fontweight='bold')
plt.ylabel('Type',fontsize=10,fontweight='bold')
plt.xlabel('Rating',fontsize=10,fontweight='bold')
plt.xticks(fontsize=10,fontweight='bold')
plt.yticks(fontsize=10,fontweight='bold');


# #### Cost of Restuarant

# In[86]:


sns.countplot(zomato['cost'])
sns.countplot(zomato['cost']).set_xticklabels(sns.countplot(zomato['cost']).get_xticklabels(), rotation=90, ha="right")
fig = plt.gcf()
fig.set_size_inches(15,15)
plt.title('Cost of Restuarant')


# #### No. of Restaurants in a Location

# In[87]:


fig = plt.figure(figsize=(20,7))
loc = sns.countplot(x="location",data=zomato_real, palette = "Set1")
loc.set_xticklabels(loc.get_xticklabels(), rotation=90, ha="right")
plt.ylabel("Frequency",size=15)
plt.xlabel("Location",size=18)
loc
plt.title('NO. of restaurants in a Location',size = 20,pad=20)


# #### Restaurant type

# In[88]:


fig = plt.figure(figsize=(17,5))
rest = sns.countplot(x="rest_type",data=zomato_real, palette = "Set1")
rest.set_xticklabels(rest.get_xticklabels(), rotation=90, ha="right")
plt.ylabel("Frequency",size=15)
plt.xlabel("Restaurant type",size=15)
rest 
plt.title('Restaurant types',fontsize = 20 ,pad=20)


# #### Most famous Restaurant chains in Bengaluru

# In[89]:


plt.figure(figsize=(15,7))
chains=zomato_real['name'].value_counts()[:20]
sns.barplot(x=chains,y=chains.index,palette='Set1')
plt.title("Most famous restaurant chains in Bangaluru",size=20,pad=20)
plt.xlabel("Number of outlets",size=15)


# ### Linear Regression

# In[49]:


#Prepare a Linear Regression Model
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# ### Decision Tree Regression 

# In[50]:


#Prepairng a Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=105)
DTree=DecisionTreeRegressor(min_samples_leaf=.0001)
DTree.fit(x_train,y_train)
y_predict=DTree.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_predict)


# ### Random Forest Regression

# In[51]:


#Preparing Random Forest REgression
from sklearn.ensemble import RandomForestRegressor
RForest=RandomForestRegressor(n_estimators=500,random_state=329,min_samples_leaf=.0001)
RForest.fit(x_train,y_train)
y_predict=RForest.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_predict)


# ### Extra Tree Regressor

# In[52]:


#Preparing Extra Tree Regression
from sklearn.ensemble import  ExtraTreesRegressor
ETree=ExtraTreesRegressor(n_estimators = 100)
ETree.fit(x_train,y_train)
y_predict=ETree.predict(x_test)


from sklearn.metrics import r2_score
r2_score(y_test,y_predict)


# In[53]:


import pickle
# Saving model to disk
pickle.dump(ETree, open('model.pkl','wb'))


# It can be observed that we have got the best accuracy for Extra tree regressor

# In[ ]:




