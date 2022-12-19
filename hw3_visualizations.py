import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer

ps3_patient_zet = np.load('ps3_patient_zet.npy')
X = np.load('ps3_genetic_fingerprints.npy')
# df1 = pd.DataFrame(ps3_patient_zet)
df2 = pd.DataFrame(X)

sil = []
fig, ax = plt.subplots(5, 2, figsize=(15,8))
kmax = 11
for i in range(2, kmax+1):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=100, n_init=10, random_state=42)
    kmeans.fit(df2)
    labels = kmeans.labels_
    sil.append(silhouette_score(df2, labels, metric = 'euclidean'))
    #wcss.append(kmeans.inertia_)

    q, mod = divmod(i, 2)
    visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(df2)

plt.show()

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

ks = range(2, 11)
inertias = []
sil = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    # Fit model to samples
    model.fit(X)
    y_pred = model.predict(X)
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    sil.append((silhouette_score(X, y_pred)))
    
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

plt.plot(ks, sil, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

model = PCA(n_components=2)
results = model.fit_transform(df2)
plt.plot(results[:, 0], results[:, 1], 'k.', markersize=2)
plt.show()
print(results.shape)

# inertia = []
# for k in range(2, 11):
#     kmeans = KMeans(n_clusters=k, random_state=1).fit(X)
#     inertia.append(np.sqrt(kmeans.inertia_))

# plt.plot(range(2, 11), inertia, marker='s');

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df2)

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)
plt.show()

PCA_components = pd.DataFrame(principalComponents)

# print(PCA_components)

plt.scatter(PCA_components[0], PCA_components[1], alpha=.1, color='black')
plt.xlabel('PCA 0')
plt.ylabel('PCA 1')
plt.show()
km = KMeans(n_clusters=5)
km.fit(X)
y_pred = km.predict(ps3_patient_zet.reshape(1,-1))

print(y_pred)
"""