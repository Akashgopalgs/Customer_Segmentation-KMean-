#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv(r"C:\Users\akash\OneDrive\Documents\dataset\kaggle dataset\Customer Segmentation.zip")
df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# There are 24 missing values in "Income" variable. We will replace it with its median.

# In[6]:


df['Income'].fillna(df['Income'].median(),inplace=True)


# In[7]:


df.isnull().sum()


# "Dt_Customer" indicates the day particular customer registered with the firm. Check the newest and oldest recorded dates.

# In[8]:


df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'],format ="%d-%m-%Y")
dates =[]

for i in df['Dt_Customer']:
    i = i.date()
    dates.append(i)

print("The newest customer's enrolment date in therecords:",max(dates))
print("The oldest customer's enrolment date in the records:",min(dates))


# Explore the unique values in the categorical features to get a clear idea of the data.
# 

# In[9]:


print("Total categories in the feature Marital_Status:\n", df["Marital_Status"].value_counts(), "\n")
print("Total categories in the feature Education:\n", df["Education"].value_counts())


# - Calculate "Age" of a customer by the "Year_Birth".(indicating the birth year of the respective person.) We will calculate age till 2014. As we have data till 2014.

# In[10]:


df['Age_on_2014'] = 2014-df['Year_Birth']


# Add new feature "Spent" indicating the total amount spent by the customer in various categories over the span of two years.

# In[11]:


df['Spent'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']


# Created new feature "Living_with" out of "Marital_Status" to extract the living situation of couples.

# In[12]:


df['Living_with'] = df['Marital_Status'].replace({"Married":"Partner", "Together":"Partner", "Absurd":"Alone", "Widow":"Alone", "YOLO":"Alone", "Divorced":"Alone", "Single":"Alone"})


# Add new feature "Children" to indicate total children in household (kid + Teen).

# In[13]:


df['Children'] = df['Kidhome'] + df['Teenhome']


# Add new feature "Family_size".

# In[14]:


df['Family_size'] = df['Living_with'].replace({'Alone':1, "Partner": 2})+ df['Children']


# Add new feature "Is_parent" indicating parenthood status.

# In[15]:


df['Is_parent'] = np.where(df.Children>0,1,0)


# Simplify values of "Education". (In three groups)

# In[16]:


df['Education'] = df['Education'].replace({"Basic":"Undergraduate", "2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})


# In[17]:


df =df.rename(columns={"MntWines": "Wines","MntFruits":"Fruits","MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})


# Drop Extra features.

# In[18]:


df = df.drop(columns = ['Marital_Status', 'Dt_Customer', 'ID', 'Year_Birth', 'Z_CostContact', 'Z_Revenue'], axis = 1)


# In[19]:


df.head()


# In[20]:


df.describe()


# Clearly, there are few outliers in the "Income" and "Age_on_2014" feature. We will drop the outliers by setting a cap on Age and Income.

# In[21]:


df = df[(df["Age_on_2014"]<90)]
df = df[(df["Income"]<600000)]
print("The total number of data-points after removing the outliers are:", len(df))


# In[22]:


df.columns


# Check Distribution of Data

# In[23]:


plt.figure(figsize = (12, 12))
plt.subplots_adjust(hspace = 1.5, wspace=0.5)

plt.subplot(5, 3, 1)
sns.histplot(df, x = 'Income', kde = True, bins = 20)
plt.title("Distribution of Income")

plt.subplot(5, 3, 2)
sns.histplot(df, x = 'Recency', kde = True, bins = 20)
plt.title("Distribution of Recency")

plt.subplot(5, 3, 3)
sns.histplot(df, x = "Wines", kde = True, bins = 20)
plt.title("Distribution of Amount spent on Wines")

plt.subplot(5, 3, 4)
sns.histplot(df, x = 'Fruits', kde = True, bins = 20)
plt.title("Distribution of amount spent on Fruits")

plt.subplot(5, 3, 5)
sns.histplot(df, x = 'Meat', kde = True, bins = 20)
plt.title("Distribution of amount spent on Meat")

plt.subplot(5, 3, 6)
sns.histplot(df, x = 'Fish', kde = True, bins = 20)
plt.title("Distribution of amount spent on Fish")

plt.subplot(5, 3, 7)
sns.histplot(df, x = 'Sweets', kde = True, bins = 20)
plt.title("Distribution of amount spent on Sweets")

plt.subplot(5, 3, 8)
sns.histplot(df, x = 'Gold', kde = True, bins = 20)
plt.title("Distribution of amount spent on Gold")

plt.subplot(5, 3, 9)
sns.histplot(df, x = 'NumDealsPurchases', kde = True, bins = 20)
plt.title("Distribution of Deal Purchased")

plt.subplot(5, 3, 10)
sns.histplot(df, x = 'NumWebPurchases', kde = True, bins = 20)
plt.title("Distribution of from Web Purchase")

plt.subplot(5, 3, 11)
sns.histplot(df, x = 'NumCatalogPurchases', kde = True, bins = 20)
plt.title("Distribution of from Catalog Purchase")

plt.subplot(5, 3, 12)
sns.histplot(df, x = 'NumStorePurchases', kde = True, bins = 20)
plt.title("Distribution of from Store Purchase")

plt.subplot(5, 3, 13)
sns.histplot(df, x = 'NumWebVisitsMonth', kde = True, bins = 20)
plt.title("Distribution of Visit per Month")

plt.subplot(5, 3, 14)
sns.histplot(df, x = 'Age_on_2014', kde = True, bins = 20)
plt.title("Distribution of Customer Age")

plt.subplot(5, 3, 15)
sns.histplot(df, x = 'Spent', kde = True, bins = 20)
plt.title("Distribution of amount spent")

plt.show()


# In[24]:


plt.figure(figsize = (12, 12))
plt.subplots_adjust(hspace = 1.5, wspace=0.5)

plt.subplot(5, 3, 1)
sns.histplot(df, x = 'Children', kde = True, bins = 20)
plt.title("Distribution of number of Children")

plt.subplot(5, 3, 2)
sns.histplot(df, x = 'Family_size', kde = True, bins = 20)
plt.title('Distribution of Family Size')

plt.subplot(5, 3, 3)
sns.countplot(df, x = 'Education')
plt.title("Distribution of Education Level")

plt.subplot(5, 3, 4)
sns.countplot(df, x = 'AcceptedCmp1')
plt.title("Distribution of accepted Cmp1")

plt.subplot(5, 3, 5)
sns.countplot(df, x = 'AcceptedCmp2')
plt.title("Distribution of accepted Cmp2")

plt.subplot(5, 3, 6)
sns.countplot(df, x = 'AcceptedCmp3')
plt.title("Distribution of accepted Cmp3")

plt.subplot(5, 3, 7)
sns.countplot(df, x = 'AcceptedCmp4')
plt.title("Distribution of accepted Cmp4")

plt.subplot(5, 3, 8)
sns.countplot(df, x = 'AcceptedCmp5')
plt.title("Distribution of accepted Cmp5")

plt.subplot(5, 3, 9)
sns.countplot(df, x = 'Complain')
plt.title('Distribution of Complain')

plt.subplot(5, 3, 10)
sns.countplot(df, x = 'Response')
plt.title('Distribution of customer responded')

plt.subplot(5, 3, 11)
sns.countplot(df, x = 'Living_with')
plt.title("Distribution of customer (single/couple)")

plt.subplot(5, 3, 12)
sns.countplot(df, x = 'Is_parent')
plt.title("Distribution of customer (parent or not)")

plt.show()


# From Distribution Plots, we can see that, Majority features are Right-skewed except Recency, Age and Family_size.

# In[25]:


plt.figure(figsize = (12, 6))
sns.lineplot(df, x = 'Age_on_2014', y = 'Spent')
plt.title("Spent vs Age")
plt.show()
print(f"\nCorrelation between Age_on_2014 and Spent: {df['Age_on_2014'].corr(df['Spent'])}")


# From Spent vs Age plot, we can see that as Age increase, Spent also increases. Correlation score is 0.114.

# In[26]:

#
# plt.figure(figsize = (12, 6))
# sns.scatterplot(df, x = 'Income', y = 'Spent')
# plt.title("Spent vs Income")
# plt.grid(False)
# plt.show()
# print(f"\nCorrelation between Age_on_2014 and Spent: {df['Income'].corr(df['Spent'])}")


# Convert Categorical variables into Numerical Variables.

# In[27]:


# Check how many categorical variables are present in data
a = (df.dtypes == 'object')
object_cols = list(a[a].index)
print("Categorical variables in the dataset:", object_cols)


# In[28]:


# Convert to numerical using LabelEncoder
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()
for i in object_cols:
  df[i] = df[[i]].apply(LE.fit_transform)


# In[29]:


df.head()


# Check correlation amongst the features.

# In[30]:


corrmax = df.corr()
plt.figure(figsize = (25, 20))
sns.heatmap(corrmax, annot = True, cmap = 'coolwarm', center = 0)
plt.show()


# In[31]:


# Create a copy of data
df1 = df.copy()


# Scale the data

# In[32]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df1)
scaled_df1 = pd.DataFrame(scaler.transform(df1), columns = df1.columns)


# In[33]:


scaled_df1


# ## Clustering

# #### Choosing the number of cluster using Elbow method

# We will use KElbowVisualizer

# In[34]:


from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

print("Elbow Method to determine the number of clusters to be formed:")
elbow = KElbowVisualizer(KMeans(), k = 10)
elbow.fit(scaled_df1)
elbow.show()


# The inertia is not a good performance metric when trying to choose K. Because it keeps getting lower as we increase K. Indeed, the more clusters there are, the closer each instance will be to its closest centroid, and therefore the lower the inertia will be. This technique for choosing the best value for the number of clusters is rather coarse.
# 
# A more precise (but also more computationally expensive) approach is to use the silhouette score.

# **Near +1**: The sample is far away from the neighboring clusters, indicating it is well-clustered.
# 
# **0**: The sample is on or very close to the decision boundary between two neighboring clusters.
# 
# **Near -1**: The sample may be assigned to the wrong cluster.
# 
# The number of clusters that maximizes the silhouette score is often considered the best choice.

# In[35]:


from sklearn.metrics import silhouette_score

kmeans_per_k = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(scaled_df1)
                for k in range(2, 11)]

silhouette_scores = [silhouette_score(scaled_df1, model.labels_)
                     for model in kmeans_per_k[1:]]

plt.figure(figsize=(8, 3))
plt.plot(range(2, 10), silhouette_scores, "bo-")
plt.xlabel("$k$")
plt.ylabel("Silhouette score")

plt.grid(True)

plt.show()


# In[36]:


cluster_range = range(2, 10)
for i, score in zip(cluster_range, silhouette_scores):
  print(f"Silhouette Score for {i} Clusters:", score)


# Silhouette score for 2 and 3 Clusters are nearly same (0.184)
# 
# Silhouette score for 5 Clusters are 0.132
# 

# ---
# 
# 
# 
# An even more informative visualization is given when you plot every instance's silhouette coefficient, sorted by the cluster they are assigned to and by the value of the coefficient. This is called a silhouette diagram. Each diagram contains one knife shape per cluster. The shape's height indicates the number of instances in the cluster, and its width represents the sorted silhouette coefficients of the instances in the cluster (wider is better).
# 
# The vertical dashed lines represent the mean silhouette score for each number of clusters. When most of the instances in a cluster have a lower coefficient than this score, then the cluster is rather bad since this means its instances are much too close to other clusters.

# In[37]:


from sklearn.metrics import silhouette_samples
from matplotlib.ticker import FixedLocator, FixedFormatter

plt.figure(figsize=(11, 10))

for k in (2, 3, 4, 5):
    plt.subplot(4, 2, k - 1)

    y_pred = kmeans_per_k[k - 1].labels_
    silhouette_coefficients = silhouette_samples(scaled_df1, y_pred)

    padding = len(scaled_df1) // 30
    pos = padding
    ticks = []
    for i in range(k):
        coeffs = silhouette_coefficients[y_pred == i]
        coeffs.sort()
        
        color = plt.cm.Spectral(i / k)
        plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ticks.append(pos + len(coeffs) // 2)
        pos += len(coeffs) + padding

    plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
    plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
    if k in (3, 5):
        plt.ylabel("Cluster")

    if k in (5, 6):
        plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.xlabel("Silhouette Coefficient")
    else:
        plt.tick_params(labelbottom=False)
        
    plt.axvline(x=silhouette_scores[k - 2], color="red", linestyle="--")
    plt.title(f"$k={k}$")

plt.show()


# In[38]:


kmean_2 = KMeans(n_clusters=2,n_init=10,random_state=4)
kmean_2.fit(scaled_df1)


# In[39]:


y_Kmean = kmean_2.labels_


# In[40]:


kmean_2.labels_


# In[51]:


scaled_data = scaled_df1.to_numpy()
#
# k_values = [2, 3]  # Clusters 2 and 3
#
# for i in k_values:
#     # Initialize and fit KMeans
#     kmean = KMeans(n_clusters=i, n_init=10, random_state=4)
#     kmean.fit(scaled_data)
#
#     # Get the cluster labels and centroids
#     y_kmean = kmean.labels_
#     centroids = kmean.cluster_centers_
#
#     # Plot the data points, color-coded by cluster
#     plt.figure(figsize=(8, 6))
#     for cluster in range(i):
#         cluster_data = scaled_data[y_kmean == cluster]
#         plt.scatter(cluster_data[:, 1], cluster_data[:, 4], label=f'Cluster {cluster + 1}')
#
#     # Plot the centroids
#     plt.scatter(centroids[:, 1], centroids[:, 4], s=300, c='yellow', marker='X', label='Centroids')
#
#     # Add title and legend
#     plt.title(f'KMeans Clustering with {i} Clusters')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.legend()
#     plt.show()


# In[48]:





# #### Apply KMeans algorithm with K=2

# In[44]:


kmeans = KMeans(n_clusters= 2, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_df1)
df1['Cluster'] = cluster_labels
# df1.to_excel('Clustered_data.xlsx', index = False)


# In[45]:


df1.head()


# In[52]:


df['Cluster'] = cluster_labels
df.head()


# #### Number of datas distributed in clusters

# In[53]:


cluster_distribution = df['Cluster'].value_counts().sort_index()

plt.bar(cluster_distribution.index, cluster_distribution.values)
plt.xlabel('Cluster')
plt.ylabel('Number of Data Points')
plt.title('Distribution of Data Points Across Clusters')
plt.show()


# #### Analysis of Clusters
# 
# #### Cluster's Profile based on Income and Spending

# In[54]:


sns.scatterplot(df, x = 'Spent', y = 'Income', hue = 'Cluster')
plt.title("Cluster's Profile based on Income and Spending")
plt.legend()
plt.show()


# ### Cluster Pattern
# 
# - Cluster 1: High spending and High Income
# - Cluster 0: Low spending and average to low income

# #### Boxplot for spent according to cluster

# In[55]:


plt.figure(figsize = (12, 6))
sns.boxenplot(df, x = 'Cluster', y = 'Spent')

plt.show()


# ##### Explore how did our campaigns do in the past

# In[56]:


df['Total_promos'] = df['AcceptedCmp1'] + df['AcceptedCmp2'] + df['AcceptedCmp3'] + df['AcceptedCmp4'] + df['AcceptedCmp5']

plt.figure(figsize = (12, 6))
sns.countplot(df, x = 'Total_promos', hue = 'Cluster')
plt.title("Count of Promotion Accepted")
plt.xlabel("Number of Total Accepted Promotions")
plt.show()


# There has not been an overwhelming response to the campaigns so far. Very few participants overall. Moreover, no one take part in all 5 of them. Perhaps better targeted and well-planned campaigns are required to boost sales.

# In[57]:


Personal = ['Kidhome', 'Teenhome', 'Age_on_2014', 'Children', 'Family_size', 'Is_parent', 'Education', 'Living_with']

# for i in Personal:
#   plt.figure(figsize = (12, 6))
#   sns.jointplot(x = df[i], y = df['Spent'], hue = df['Cluster'], kind = 'kde')
#   plt.show()


# In[ ]:



# label_encoders = {}
# for col in df.select_dtypes(include=['object']).columns:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])
#     label_encoders[col] = le
# # In[ ]:
# import pickle
#
# preprocessed_data_path = 'preprocessed_customer_data.pkl'
# with open(preprocessed_data_path, 'wb') as file:
#     pickle.dump(df, file)
#
# with open('customer_segmentation_model.pkl', 'wb') as file:
#     pickle.dump({
#         'model': kmeans,
#         'scaler': scaler,
#         'label_encoders': label_encoders,
#         'features': df.drop('Cluster', axis=1).columns.tolist()
#     }, file)
import pickle
import warnings
warnings.filterwarnings("ignore")

df=df.drop(columns='Cluster')

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Scaling the data
scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Clustering using KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_df)

# Save the preprocessed data and model
with open('preprocessed_customer_data.pkl', 'wb') as file:
    pickle.dump(df, file)

with open('customer_segmentation_model.pkl', 'wb') as file:
    pickle.dump({
        'model': kmeans,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'features': df.drop('Cluster', axis=1).columns.tolist()
    }, file)