#!/usr/bin/env python
# coding: utf-8

# # <i>End to End Machine Learning Project</i>

# 1. Download the data.
# 2. Load the data
# 3. Visualize each attribute from loaded data.
# 4. Create a test data using StratifiedShuffleSplit.
# 5. Calculate standard correlation coefficient to know the required attributes.(0,1,-1) => 0 independent, 1,-1=> dependent.
# 6. Remove predicted attribute to prepare the data for ML algorithms.
# 7. Data cleaning=> 
# 	7.1 Numerical values clean, => SimpleImputer
# 	7.2 convert Text attributes to categorical 	values(numerical). => OneHotEncoder
# 8. Feature Scaling => Transform all the attribute values into same scale. => StandardScaler.
# 9. Transformation Pipelines => combines numerical, categorical attributes. => ColumnTransformer.
# 10. Select the Models, check the error rate and check whether overfitting or underfitting and select the models. (2 to 5).
# 11. Fine tune the model
# 12. Evaluate the System on the Test Set
# 13. Launch, Monitor, and Maintain the System

# # 1. Download the data

# Downloading the data from available source and writing script to extract the data from tgz to csv
# It is useful when the data is changed regularly, then it is better to write a script and whenever the data changes
# then I can run the script so that the data will extract and can fetch latest data.
# To do this automatically can be done through Job scheduling.
# It also helps if needed to run in multiple machines.

# In[1]:


import os
import tarfile
from six.moves import urllib


# In[2]:


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


# In[3]:


def fetch_housing_data(housing_url=HOUSING_URL,housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path,"housing.tgz")
    urllib.request.urlretrieve(housing_url,tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
fetch_housing_data()


# The data is stored in This PC>C:>Users>venis>datasets>housing>housing.csv

# # 2. Load the data

# Now lets load the data using pandas

# In[4]:


import pandas as pd


# In[5]:


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)


# In[6]:


df_housing = load_housing_data()
df_housing


# In[7]:


df_housing.count()


# In[8]:


df_housing.head() #to get top 5 rows from data frame


# In[9]:


df_housing.info()


# The districts are categorized into values Ocean_proximity
# Now to know the catgories and the district counts of each category.

# In[10]:


df_housing["ocean_proximity"].value_counts()


# To get the statistics for each field in the dataframe.
# This describe method calculates only for numerical values.

# In[11]:


df_housing.describe()


# # 3. Visualize each attribute from loaded data

# A histogram for each numerical attribute

# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
df_housing.hist(bins=50, figsize=(20,15))
plt.show()


# # 4. Create Test data

# First we have to set some ratio of data aside for testing.
# steps to create test data:
# 1. Shuffle the indices of the data randomly by applying permutation.
# 2. To get length/size of the test data, multiply the length of the data to ratio of the test data.
# 3. assign the length for train set and test set
# 4. return the index locations of test and train data set.

# Data Snooping bias:
# while keeping aside, our human brains do think to train the
# data according to test data and get intersting pattern based on that
# and that leads to selecting Machine learning model. And while
# estimating generalization error set, the estimate will be almost
# equal on the test set, train set and when we deploy it into production the error rate(generalization error) varies than expected.
# This kind of thinking to train the data according to test data is
# is called Data Snooping bias.

# In[13]:


import numpy as np
def split_train_test(data, test_ratio):
    shuffle_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)* test_ratio)
    test_indices = shuffle_indices[:test_set_size]
    train_indices = shuffle_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[14]:


train_set, test_set = split_train_test(df_housing, 0.2)
len(test_set)


# In[15]:


len(train_set)


# In[16]:


from zlib import crc32


# In[17]:


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32


# In[18]:


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# Unfortunetly housing data doesnot have identifier column, so using Row index as ID it adds new column as index, else use
# ID = (longitude*1000 + latitude)

# In[19]:


housing_with_id = df_housing.reset_index()
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
housing_with_id["id"] = df_housing["longitude"] * 1000 + df_housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")


# In[20]:


len(test_set)


# Same as split_train_test process but using scikit-learn model

# In[21]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df_housing, test_size=0.2, random_state=42)


# In[22]:


len(test_set)


# # 5. Stratify sampling based on the income category

# This step is about categorizing median income for reducing sampling bais in test data.

# In[23]:


df_housing["income_category"] = pd.cut(df_housing["median_income"], bins=[0.,1.5,3.0,4.5,6., np.inf],
                                      labels = [1,2,3,4,5])
df_housing["income_category"].hist()


# In this step, Take 20% from each category as test data then the data is stratified sampling based on the income category.
# No sampling bais in test data.
# For this use Scikit-Learn’s StratifiedShuffleSplit class.

# In[24]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(df_housing, df_housing["income_category"]):
    strat_train_set = df_housing.loc[train_index]
    strat_test_set = df_housing.loc[test_index]


# In[25]:


len(strat_test_set)


# In[26]:


df_housing["income_category"].value_counts()


# 20% of each category from whole dataset of income_category

# In[27]:


strat_test_set["income_category"].value_counts()


# percentage of each category from median income in test set

# In[28]:


strat_test_set["income_category"].value_counts()/len(strat_test_set)


# In[29]:


df_housing


# To drop income_category

# In[30]:


# for set_ in (strat_train_set, strat_test_set):
#     set_.drop("income_category", axis=1, inplace=True)


# # TEST SET

# In[31]:


strat_test_set


# # WHOLE SET

# In[32]:


df_housing


# # TRAIN SET

# In[33]:


housing_train_set = strat_train_set.copy()
housing_train_set


# # 6. A geographical scatterplot of the data(train set)

# In[34]:


housing_train_set.plot(kind="scatter",x="longitude",y="latitude")


# Dense areas look darker, sparse areas look lighter

# In[35]:


housing_train_set.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1)


# s = The radius of each circle which represents the district’s population.
# 
# c = The color represents the price. 
# 
# jet in cmap = Predefined color map.
# 
# color ranges = blue to red.
# 
# blue = low median housing prices.
# 
# red = high median housing prices.

# In[36]:


housing_train_set.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,s=housing_train_set["population"]/100,
                      label= "population", figsize=(10,7), c="median_house_value", cmap=plt.get_cmap("jet"),colorbar=True)
plt.legend()


# # 7. standard correlation coefficient (also called Pearson’s r)

# correlation = How strongly are two attributes related to each other.
# 
# coefficient = A coefficient is just a number that multiplies something.
# 
# correlation coefficient = Number that tells you how strong and in what direction two attributes (variables) are related.
# 
# standard correlation coefficient(Pearson’s correlation coefficient) = The value of Pearson’s r ranges between:
# 
# +1 = perfect positive linear relationship
# 
# (e.g., as one goes up, the other always goes up)
# 
# 0 = no linear relationship
# 
# (e.g., the two are totally unrelated)
# 
# −1 = perfect negative linear relationship
# 
# (e.g., as one goes up, the other always goes down)
# 
# Formula => r = cov(X, Y) / (std(X) * std(Y))
# 
# cov(X, Y) = (sum(x - mean(x))sum(y-mean(y)))/sqrt(sum(x - mean(x))^2 * sum(y - mean(y))^2)
# 
# x - Each individual value of variable X
# 
# 𝑦 - Each individual value of variable Y
# 
# mean(x) - Mean (average) of all X values
# 
# mean(𝑦) - Mean (average) of all Y values
# 
# Summation (add up all values)
# 

# In[37]:


corr_matrix = housing_train_set.corr()


# In[38]:


corr_matrix["median_house_value"].sort_values(ascending=False)


# # 8. Scatter matrix of standard correlation coefficient

# if standard correlation coefficient is 1 => linear correlation goes up for up, down for down.
# 
# if standard correlation coefficient is -1 => linear correlation goes up for down, down for up.
# 
# if standard correlation coefficient is 0 => Nonlinear correlation(attributes not depending on each other).

# scatter_matrix function => which plots every numerical attribute against every other numerical attribute.
# Since there are now 11 numerical attributes, would get 11^2 = 121 plots, 
# we will plot for only related attributes.

# In[39]:


from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income","total_rooms","housing_median_age"]
scatter_matrix(housing_train_set[attributes], figsize=(12,8))


# Median income versus median house value

# In[40]:


housing_train_set.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)


# the total number of rooms in a district is not very useful if you don’t know how many households there are. 
# 
# What you really want is the number of rooms per household.
# 
# Similarly, the total number of bedrooms by itself is not very useful, What to want is the number of bedrooms per rooms. 
# 
# similarly the population is not useful, what to want is the population per household.

# In[41]:


housing_train_set["rooms_per_household"] = housing_train_set["total_rooms"]/housing_train_set["households"]
housing_train_set["bedrooms_per_room"] = housing_train_set["total_bedrooms"]/housing_train_set["total_rooms"]
housing_train_set["population_per_household"] = housing_train_set["population"]/housing_train_set["households"]


# In[42]:


corr_matrix = housing_train_set.corr()


# now compare median_house_value vs total_bedrooms and median_house_value vs bedrooms_per_room, then we got some relation instead of 0.04(correlation = 0)

# In[43]:


corr_matrix["median_house_value"].sort_values(ascending=False)


# # 9. Prepare the Data for Machine Learning Algorithms

# To separate the predictors and the labels
# 
# 1. drop() =>  creates a copy of the data and does not affect strat_train_set, housing_drop_set doesnot have median_house_value.
# 
# 2.  median_house_value labels assigned to housing_labels, housing_labels contains median_house_value.

# In[44]:


housing_drop_set = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
housing_labels


# In[45]:


housing_drop_set


# # 10. DATA CLEANING

# In our dataset, total_bedrooms attribute has missing values, to get rid of that we have 3 options
# 1. Get rid of the corresponding districts.
# 2. Get rid of the whole attribute.
# 3. Set the values to some value (zero, the mean, the median, etc.)

# You can accomplish these easily using DataFrame’s dropna(), drop(), and fillna() methods:
# 
# housing_drop_set.dropna(subset=["total_bedrooms"])     # option 1
# 
# housing_drop_set.drop("total_bedrooms", axis=1)       # option 2
# 
# median = housing["total_bedrooms"].median()  # option 3
# housing_drop_set["total_bedrooms"].fillna(median, inplace=True)

# It is hard to use any of the above 3 methods, once the system goes live to replace missing values in new data.
# for that we use Scikit learn's SimpleImputer class.

# In[46]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')


# Since the median can only be computed on numerical attributes, we need to create a copy of the data without the text attribute ocean_proximity

# In[47]:


housing_without_op = housing_drop_set.drop("ocean_proximity", axis=1)


# Now fit the imputer instance to the training data using the fit() method

# In[48]:


imputer.fit(housing_without_op)


# The imputer has simply computed the median of each attribute and stored the result in its statistics_ instance variable.

# In[49]:


imputer.statistics_


# In[50]:


housing_without_op.median().values


# In[51]:


transformed_data = imputer.transform(housing_without_op)


# Converting transformed_data (NumPy array) to housing_trained_cleaned (Pandas data frame)

# In[52]:


housing_trained_cleaned = pd.DataFrame(transformed_data, columns=housing_without_op.columns)


# In[53]:


housing_trained_cleaned


# In[54]:


housing_trained_cleaned.info()


# # Scikit-Learn Design

# The main design principles

# Consistency: All objects share a consistent and simple interface.
# 
#     1. Estimators: Any object that can estimate some parameters based on the dataset.
#     eg: imputer is an estimator
#         fit() method through estimation performed.
#         fit() method takes dataset as a parameter to fit into the system.
#         imputer's strategy is the parameters(hyperparameters).
#         
#      2. Transformers: Estimators which transform the dataset is called Transformers.
#      The transformation is performed by tranform() method, parameter as dataset.
#      It returns transformed dataset.
#      Also have fit_transform() => fit() + transform(), it is optimized and runs much faster.
#     
#     3. Predictors: Finally these estimators are capable of making predictions by a dataset.
#     eg: LinearRegression is the predictor.
#     
#     Predict() method takes the dataset of new instances and returns the corresponding predictions.
#     Also have a score() method that measures the quality of the predictions for a test set.
#     
# Inspection: 
#     All estimator's hyperparameters are accessible directly public instance variables.
#     
#     eg: imputer.strategy.
#     
#     All estimator's learned parameters are also accessible directly public instance variables with an underscore suffix.
#     eg: imputer.statistics_
#     
# Nonproliferation of classes: 
#     Datasets are represented as NumPy arrays or SciPy sparse matrices.
#     
#     Hyperparameters are Python strings or numbers.
#     
# Composition: Create a pipeline estimator from an arbitary sequence of transformers followed by final estimator and It can be reused.
# 
# Sensible defaults: Scikit-Learn provides reasonable default values for most parameters.
# 
# 
# 

# # Handling Text and Categorical Attributes

# Earlier we left out the categorical attribute ocean_proximity because it is a text attribute so we cannot compute its median

# In[55]:


housing_category = housing_drop_set[["ocean_proximity"]]


# In[56]:


housing_category


# Most ML algorithms prefer to work with numerical values, so lets convert ocean_proximity values to numbers. For this we use 
# 
# <b>Scikit-Learn’s OrdinalEncoder class</b>

# In[57]:


from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_category_encoded = ordinal_encoder.fit_transform(housing_category)
housing_category_encoded


# to get the categories of housing_category_encoded values, it returns in 1D array since the dataset contains only one categorical attribute.

# In[58]:


ordinal_encoder.categories_


# One issue with ordinal encoder, 
# ML algorithms will assume that two nearby values are more similar than two distant values.
# 
# This may be fine in some cases (e.g., for ordered categories such as “bad”, “average”, “good”, “excellent”),
# but it is obviously not the case for the ocean_proximity column (for example, categories 0 and 4 are clearly more similar than categories 0 and 1).

# To fix this issue, a common solution is to create one binary attribute per category.
# one attribute equal to 1 when the category is “<1H OCEAN” (and 0 otherwise), 
# another attribute equal to 1 when the category is “INLAND” (and 0 otherwise), and so on.
# 
# This is called <b>one-hot Encoding</b>, because only one attribute will be equal to 1 (hot), while the others will be 0 (cold). 

# In[59]:


from sklearn.preprocessing import OneHotEncoder
category_encoder = OneHotEncoder()
housing_category_1hot = category_encoder.fit_transform(housing_category)
housing_category_1hot


# output is a SciPy sparse matrix,  This is very useful when you have categorical attributes with thousands of categories. After one hot encoding we get a matrix with thousands of columns, and the matrix is full of zeros except for a single 1 per row. 
# so it is wasteful of store zeros. You can use it mostly like a normal 2D array.

# In[60]:


housing_category_1hot.toarray()


# In[61]:


category_encoder.categories_


# <b>Representation Learning: </b>
# Instead of using onehotencoder, calculate ocean_proximity as distance between ocean and district(block groups)
# Replace each category with a learnable low dimensional vector called an embedding.

# In[62]:


housing_trained_cleaned


# <b>Custom transformers: </b>
# For custom cleanup operations or combining specific attributes.

# In[63]:


from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
attr_addre = CombinedAttributesAdder(add_bedrooms_per_room = False)
housing_extra_attrins = attr_addre.transform(housing_drop_set.values)


# # Feature Scaling

# Machine Learning algorithms don’t perform well when the input numerical attributes have very different scales.
# 
# This is the case for the housing data: the total number of rooms ranges from about 6 to 39,320, while the median
# incomes only range from 0 to 15.
# 
# The 2 common ways to get all attributes to have same scale are <b>min-max scaling, standardization.</b>
# 
# Mix-Max scaling/Normalization:  values are shifted and rescaled so that they end up ranging from 0 to 1.
#     x-min/max-min
# Scikit-Learn provides a transformer called MinMaxScaler. It has a feature range, can give custom range.
# 
# Standardization: x-(mean value)/standard deviation.
# Scikit-Learn provides a transformer called StandardScaler.
# 
# Standardization is much less affected by outliers.
# 
# Eg: suppose a district had a median income equal to 100 (by mistake). Min-max scaling would then the values from 0–15 down to 0–0.15, whereas standardization would not be much affected.
#     

# # Transaformation Pipelines

# Scikit-Learn provides the Pipeline class to help with such sequences of transformations. 

# In[64]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])
housing_num_tr = num_pipeline.fit_transform(housing_without_op)


# In[65]:


housing_num_tr


# Till now, we have transformed the numerical columns, categorical columns seperately.
# To combine these into one transformer, Scikit-Learn introduced the <b>ColumnTransformer.</b>
# 
# 

# In[66]:


from sklearn.compose import ColumnTransformer
num_attribs = list(housing_without_op)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])
housing_prepared = full_pipeline.fit_transform(housing_drop_set)
housing_prepared


# # Select and Train a Model

# Training and Evaluating on the Training Set, Training the linear regression model.

# In[67]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
some_data = housing_drop_set.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Prediction: ",lin_reg.predict(some_data_prepared))
print("Labels: ",list(some_labels))


# Let’s measure this regression model’s RMSE on the whole training set using Scikit-Learn’s mean_squared_error function

# In[68]:


from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# most districts median_housing_values range between $120000 and $265000, so a typical prediction error of $68376.51254853733 is not very satisfying. This is an example of a model <b>underfitting</b> the training data. 
# we knew that there are 3 ways to resolve underfitting problem
# 1. To select a more powerful model.
# 2. To feed the training algorithm with better features.
# 3. To reduce the constraints on the model.

# In[69]:


# 1. To select a more powerful model
# Let’s train a DecisionTreeRegressor 
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)


# In[70]:


some_data = housing_drop_set.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Prediction: ",tree_reg.predict(some_data_prepared))
print("Labels: ",list(some_labels))


# In[71]:


housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# It is much more likely that the model has badly overfit the data.

# <b>Better Evaluation Using Cross-Validation</b>

# One way to evaluate the Decision Tree model would be to use the train_test_split function to split the training set into a smaller training set and a validation set, then train your models against the smaller training set and evaluate them against the validation set. 
# 
# Alternatively, use Scikit-Learn’s K-fold cross-validation feature.
# The following code randomly splits the training set into 10 distinct subsets called folds, then it trains and evaluates the Decision Tree model 10 times, picking a different fold for evaluation every time and training on the other 9 folds. The result is an array containing the 10 evaluation scores.
# 

# In[72]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                        scoring = "neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# Scikit-Learn’s cross-validation features expect a utility function (greater is better) rather than a cost function (lower is better), so the scoring function is actually the <b>opposite of the MSE</b> (i.e., a negative value), which is why the preceding code computes -scores before calculating the square root.

# In[73]:


def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())
display_scores(tree_rmse_scores)


# Now by seeing this mean 71374 and standard deviation +-2686 for the RMSE, it looks like it is worst than Linear Regression Model.
# 
# For this we used cross validation and we got to know this model also not correct but if there is large amount of instances in a dataset then Cross validation is not the best way to use, because it takes lot of time.

# <b>Let’s cross validate Linear Regression model just to be sure:</b>

# In[74]:


lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                        scoring = "neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# That’s right: the Decision Tree model is overfitting so badly that it performs worse than the Linear Regression model.

#  Let’s try one last model now: the <b>RandomForestRegressor.</b>

# Random Forests work by training many Decision Trees on random subsets of the features, then averaging out their predictions. Building a model on top of many other models is called <b>Ensemble Learning.</b>

# In[75]:


from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)


# In[76]:


housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[77]:


forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                        scoring = "neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


# Wow, this is much better than the other above 2 models. Random Forests look very promising. However, the score on the training set is still much lower than on the validation sets, meaning that the model is still overfitting the training set. Possible solutions for overfitting as we knew are to simplify the model, constrain it (i.e., regularize it), or get a lot more training data.
# 
# However, before going forward with overfitting in Random Forests i.e hyperparameter tuning, try out many other models from various categories of Machine Learning algorithms (several Support Vector Machines with different kernels, possibly a neural network, etc.). SHortlist 2 to 5 models.
# 

# You should save every model you experiment with, so you can come back easily to any model you want. Make sure you save both
# the hyperparameters and the trained parameters, as well as the cross-validation scores and perhaps the actual predictions as well. This will allow you to easily compare scores across model types, and compare the types of errors they make. You can easily save <b>Scikit-Learn models by using Python’s pickle module, or using sklearn.externals.joblib</b>, which is more efficient at serializing large NumPy arrays.

# # Fine-Tune Model

# The next step after selecting the models is to Fine-Tune the models.
# 1. <b>Grid Search :</b> 
# To search the hyperparameters combination values manually it is time taking and hard.
# Instead we have Scikit-Learn's GridSearchCV to search. All we need to do is tell which hyperparameters want to experiment with and what values to try out and it will evaluate all the possible combinations of hyperparameter values using cross-validation.

# In[91]:


from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators': [3,10,30,40,50,60,70], 'max_features': [2,4,6,8,10,12]},
    {'bootstrap': [False], 'n_estimators': [3,10], 'max_features': [2,3,4]},
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                          scoring='neg_mean_squared_error',
                          return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)


# In[92]:


grid_search.best_params_


# This param_grid tells Scikit-Learn to evaluate all 3 X 4=12 combinations of n_estimators and max_features hyperparameter values for dict1,
# 2*3 = 6 combinations for dict2, 
# from these the grid search will explore 12 + 6 = 18 combinations of RandomForestRegressor Hyperparameter values.
# As cv=5, it will train each model 5 times(five-fold cross validation), that will be 18 * 5 = 90 rounds of training.
# 

# In[93]:


# Also can get the best estimator directly,
grid_search.best_estimator_


# In[94]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# Now for 50282.743120413674 {'max_features': 8, 'n_estimators': 70}, is the mean value while using RandomForestRegressor model. that means it is better than 50405.39820934913, so we fine-tuned the model.
# 
# we can give more numbers for n_estimators to fine-tune the model again but it consumes more time and money so that is why we have better way which is <b>RandomizedSearch</b>

# <b>RandomizedSearch: </b>
# It evaluates a given number of random combinations by selecting a random
# value for each hyperparameter at every iteration.

# <b>Ensemble Methods</b>
# The other way to fine-tune is to combine the models that perform best.

# ## Analyze the Best Models and Their Errors

# The RandomForestRegressor can indicate the relative importance of each attribute for making accurate predictions:

# In[95]:


feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# Let’s display these importance scores next to their corresponding attribute names

# In[97]:


extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse = True)


# Here we can drop some of the less useful features, eg: one ocean_proximity category is usefull, so could try dropping others.

# # Evaluate Your System on the Test Set

# Now is the time to evaluate the final model on the test set with the predictors and the labels from the test set, run the full_pipeline to transform the data (call transform(), not fit_transform(), Do not want to fit the test set!).
# 

# In[105]:


final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis = 1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse


# In some cases 48327 generalization error will not be enough to convince to launch in production. What if it is just 0.1% better than the model currently in production? For that we need to have an idea of how precise the estimate is.
# For this, <b>compute a 95% confidence interval for the generalization error using scipy.stats.t.interval()</b>
# 

# In[100]:


from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                        loc=squared_errors.mean(),
                        scale = stats.sem(squared_errors)))


# ## Launch, Monitor, and Maintain Your System

# Perfect, Got approval to launch! Now need to get the solution ready for production, in particular by plugging the production input data sources into the system and writing tests.
# Now also need to write monitoring code to check the system’s live performance at regular intervals and trigger alerts when it drops. This is important to catch not only sudden breakage, but also performance degradation. This is quite common unless the models are regularly trained
# on fresh data(online training).
# 
# Monitoring the inputs is particularly important for online learning systems.
# Finally, Will generally want to train your models on a regular basis using fresh data to automate this process as much as possible. If not, there are very likely to refresh the model only every six months (at best), and the system’s performance may fluctuate severely over time. If the system is an online learning system, should make sure to save snapshots of its state at regular intervals so that can easily roll back to a previously working state.

# In[ ]:




