#!/usr/bin/env python
# coding: utf-8

# ### IMPORTING NECESSARY LIBRARIES BELOW

# In[ ]:


import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# ##### READING DATA SET BELOW

# In[23]:


my_data = pd.read_csv("data.csv", delimiter=",")
my_data[0:5]


# #### CREATING OUR INPUT VECTOR X

# In[24]:


X = my_data[['Tag','x', 'y', 'z']].values
X[0:4]


# #### PRE-PROCESSING TAG ATTRIBUTE SINCE IT CATEGORICAL IN NATURE

# In[25]:


from sklearn import preprocessing
le_tag = preprocessing.LabelEncoder()
le_tag.fit(['010-000-024-033','020-000-033-111','020-000-032-221','010-000-030-096'])
X[:,0] = le_tag.transform(X[:,0])


# #### OUR GOAL:Y IS ACTIVITY ATTRIBUTE.

# In[26]:


y = my_data["Activity"]
y[0:5]


# ##### WE BREAK OUR DATA SET INTO TRAINING AND TESTING DATASET BELOW AND APPLY INFORMATION GAIN METHOD ON IT FOR ATTRIBUTE SELECTION

# In[27]:


from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
postureTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
postureTree.fit(X_trainset,y_trainset)
predTree = postureTree.predict(X_testset)


# In[28]:


# from sklearn.model_selection import train_test_split
# X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
# postureTree2 = DecisionTreeClassifier(criterion="gini", max_depth = 4)
# postureTree2.fit(X_trainset,y_trainset)
# predTree = postureTree2.predict(X_testset)


# In[29]:


print (predTree [0:5])
print (y_testset [0:5])


# ### EVALUATING DECISION'S TREE ACCURACY BELOW

# In[30]:


from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))


# #### IMPORTING NECESSARY LIBRARIES FOR VISUALIZATION OF THE DECISION TREE

# In[32]:


from six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
get_ipython().run_line_magic('matplotlib', 'inline')


# In[33]:


dot_data = StringIO()
filename = "posture.png"
featureNames = my_data.columns[0:4]
targetNames = my_data["Activity"].unique().tolist()
out=tree.export_graphviz(postureTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')

