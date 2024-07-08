import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from google.colab import drive
drive.mount('/content/drive')

# Exploratory Data Analysis (EDA)

df = pd.read_csv('/content/drive/MyDrive/Mall_Customers.csv')
df.head()

df.info()

df.rename(index=str, columns={'Annual Income (k$)': 'Income',
                              'Spending Score (1-100)': 'Score'}, inplace=True)
df.head()

# Let's see our data in a detailed way with pairplot
sns.pairplot(df.drop('CustomerID', axis=1), hue='Gender', aspect=1.5)
plt.show()

# KMeans Clustering
## Elbow Analysis

from sklearn.cluster import KMeans

clusters = []
X = df.drop(['CustomerID', 'Gender'], axis=1)

for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=102).fit(X)
    clusters.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 4))
sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)
ax.set_title('Searching for Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')

# Annotate arrow
ax.annotate('Possible Elbow Point (n=3)', xy=(3, 140000), xytext=(3, 50000), xycoords='data',
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

ax.annotate('Possible Elbow Point (n=5)', xy=(5, 80000), xytext=(5, 150000), xycoords='data',
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

plt.show()

## Creating Visual Plots
### KMeans - 3 Clusters

# 3 cluster
km3 = KMeans(n_clusters=3, random_state=102).fit(X)

X['Labels'] = km3.labels_
plt.figure(figsize=(8, 4))
sns.scatterplot(x=X['Income'],
                y=X['Score'],
                hue=X['Labels'],
                palette=sns.color_palette('hls', 3))
plt.title('KMeans with 3 Clusters')
plt.show()

### KMeans - 5 Clusters

# Let's see with 5 Clusters

km5 = KMeans(n_clusters=5, random_state=102).fit(X)

X['Labels'] = km5.labels_
plt.figure(figsize=(8, 4))
sns.scatterplot(x=X['Income'],
                y=X['Score'],
                hue=X['Labels'],
                palette=sns.color_palette('hls', 5))
plt.title('KMeans with 5 Clusters')
plt.show()

from sklearn.metrics import silhouette_score

silhouette_score(X, km5.labels_)


fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(121)
sns.swarmplot(x='Labels', y='Income', data=X, ax=ax)
ax.set_title('Labels According to Annual Income')

ax = fig.add_subplot(122)
sns.swarmplot(x='Labels', y='Score', data=X, ax=ax)
ax.set_title('Labels According to Scoring History')

plt.show()

# Hierarchical Clustering
## Agglomerative
X = X.drop(columns='Labels')
from sklearn.cluster import AgglomerativeClustering

agglom = AgglomerativeClustering(n_clusters=5, linkage='complete').fit(X)

X['Labels'] = agglom.labels_
plt.figure(figsize=(8, 4))
sns.scatterplot(x=X['Income'],
                y=X['Score'],
                hue=X['Labels'],
                palette=sns.color_palette('hls', 5))
plt.title('Agglomerative with 5 Clusters')
plt.show()

## Dendrogram Associated for the Agglomerative Hierarchical Clustering
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix

dist = distance_matrix(X, X)

Z = hierarchy.linkage(dist, 'complete')

plt.figure(figsize=(18, 50))
dendro = hierarchy.dendrogram(Z, leaf_rotation=0, leaf_font_size=12, orientation='right')

## DBSCAN
from sklearn.cluster import DBSCAN
X = X.drop(columns='Labels')
X.head()

dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X)

X['Labels'] = dbscan.labels_
plt.figure(figsize=(8, 4))
sns.scatterplot(x=X['Income'],
                y=X['Score'],
                hue=X['Labels'],
                palette=sns.color_palette('hls', 5))
plt.title('DBSCAN with 5 Clusters')
plt.show()

# Wrapping it up
fig = plt.figure(figsize=(10,8))

##### KMeans #####
ax = fig.add_subplot(221)

X = X.drop(columns='Labels')

km5 = KMeans(n_clusters=5, random_state=102).fit(X)
X['Labels'] = km5.labels_
sns.scatterplot(x=X['Income'],
                y=X['Score'],
                hue=X['Labels'],
                palette=sns.color_palette('hls', 5), s=60, ax=ax)
ax.set_title('KMeans with 5 Clusters')


##### Agglomerative Clustering #####
ax = fig.add_subplot(222)

X = X.drop(columns='Labels')

agglom = AgglomerativeClustering(n_clusters=5, linkage='complete').fit(X)
X['Labels'] = agglom.labels_
sns.scatterplot(x=X['Income'],
                y=X['Score'],
                hue=X['Labels'],
                palette=sns.color_palette('hls', 5), s=60, ax=ax)
ax.set_title('Agglomerative with 5 Clusters')


##### DBSCAN Clustering #####
ax = fig.add_subplot(223)

X = X.drop(columns='Labels')

dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X)

X['Labels'] = dbscan.labels_
sns.scatterplot(x=X['Income'],
                y=X['Score'],
                hue=X['Labels'],
                palette=sns.color_palette('hls', 5))
plt.title('DBSCAN with 5 Clusters')

plt.tight_layout()
plt.show()
