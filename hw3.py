import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings(action="ignore")
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from yellowbrick.cluster import SilhouetteVisualizer
# importing libraries

# importing data
ps3_patient_zet = np.load('ps3_patient_zet.npy')
ps3_genetic_fingerprints = np.load('ps3_genetic_fingerprints.npy')
# creating dataframes
df1 = pd.DataFrame(ps3_patient_zet)
df2 = pd.DataFrame(ps3_genetic_fingerprints)
print(df2.shape)
# adding patient Z to population data
df4 = np.concatenate((df1, df2.T), axis=1)
df3 = pd.DataFrame(df4.T)
X= np.asarray(df3)
# scaling the data
scale = StandardScaler()
X = scale.fit_transform(X)

# n_clusters=11
# cost=[]
# for i in range(1,n_clusters):
#     kmean= KMeans(i)
#     kmean.fit(X)
#     cost.append(kmean.inertia_)  

# Calculating the inertia and silhouette_score¶
cost = []
sil = []
# changing the number of clusters 
for k in range(2,11):
    # for loop to plot and show ideal number of clusters
    kmean = KMeans(n_clusters=k, random_state=0)
    kmean.fit(X)
    y_pred = kmean.predict(X)
    cost.append((k, kmean.inertia_))
    sil.append((k, silhouette_score(X, y_pred)))

fig, ax = plt.subplots(1,2, figsize=(12,4))
# Plotting Elbow Curve
x_iner = [x[0] for x in cost]
y_iner  = [x[1] for x in cost]
ax[0].plot(x_iner, y_iner)
ax[0].set_xlabel('Number of Clusters')
ax[0].set_ylabel('Inertia')
ax[0].set_title('Elbow Curve')
# Plotting Silhouetter Score
x_sil = [x[0] for x in sil]
y_sil  = [x[1] for x in sil]
ax[1].plot(x_sil, y_sil)
ax[1].set_xlabel('Number of Clusters')
ax[1].set_ylabel('Silhouette Score')
ax[1].set_title('Silhouette Score Curve')

# plt.plot(cost, 'bx-')
# performing K-Means
kmean= KMeans(5)
kmean.fit(df3)
labels=kmean.labels_

clusters=pd.concat([df3, pd.DataFrame({'cluster':labels})], axis=1)
# print(clusters.head())

#dist = 1 - cosine_similarity(X)

pca = PCA(2)
pca.fit(X)
X_PCA = pca.transform(X)
print(X_PCA.shape)
print(df3.iloc[0])

x, y = X_PCA[:, 0], X_PCA[:, 1]
z = X_PCA[0]

colors = {0: 'red',
          1: 'blue',
          2: 'green', 
          3: 'yellow', 
          4: 'orange'}

names = {0: '1', 
         1: '2', 
         2: '3', 
         3: '4', 
         4: '5'}
  
df = pd.DataFrame({'x': x, 'y':y, 'label':labels}) 
groups = df.groupby('label')
# PLOTTING Clusters
fig, ax = plt.subplots(figsize=(10, 7)) 

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=7, color=colors[name],label=names[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')
ax.plot(z[0], z[1],ms=10, marker='o',color='black')
ax.legend()
ax.set_title("Cluster Visualization")
plt.show()

# for c in clusters:
#     grid= sns.FacetGrid(clusters, col='cluster')
#     grid.map(plt.hist, c)

x, y = X_PCA[:, 0], X_PCA[:, 1]
z = X_PCA[0]

colors = {1: 'blue'}

names = {1: '2'}
  
df = pd.DataFrame({'x': x, 'y':y, 'label':labels}) 
dfx = df.loc[df['label'] == 1]
groups = dfx.groupby('label')

fig, ax = plt.subplots(figsize=(10, 7)) 

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=10, color=colors[name],label=names[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')
ax.plot(z[0], z[1],ms=10, marker='o',color='black')
ax.legend()
ax.set_title("Cluster Visualization")
plt.show()

# code to find the number of people in the cluster that belongs to patient Z
datafra = df
datafra = datafra.drop([0])
datafra.drop(["x", "y"], axis = 1, inplace = True)
datafra['label'] = datafra['label'].replace([2,3,4],[0,0,0])
#print(datafra)
print(datafra['label'].value_counts())

fig, ax = plt.subplots(1, 1, figsize=(15,8))
kmax = 11
for i in range(2, kmax+1):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=100, n_init=10, random_state=42)
    kmeans.fit(df)
    labels = kmeans.labels_
    sil.append(silhouette_score(df, labels, metric = 'euclidean'))
    #wcss.append(kmeans.inertia_)

    q, mod = divmod(i, 2)
    visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(df)