import numpy as np
np.set_printoptions(threshold=10000,suppress=True)
import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score



data = pd.read_csv('villes.csv', sep=';')
X = data.iloc[:, 1:13].values
labels = data.iloc[:, 0].values

x = StandardScaler().fit_transform(X)

pca = PCA()
principalComponents = pca.fit_transform(x)
somme = 0
ite = 0
for i in pca.explained_variance_ratio_:
    if(somme >= 0.90):
        break
    else:
        somme += i
        ite += 1
print("il faut ",ite," principale compoenent pour avoir 90% de l'info")

# store the silhouette score of each of the clustering
Score = {}

X_pca = principalComponents[:,:2]

kmeans = KMeans(n_clusters=3, init='k-means++').fit(X)
clustering = kmeans.labels_

Score["kmeans_3_cluster_ori"] = silhouette_score(X, clustering,metric='euclidean')

colors = ['red','yellow','blue','pink']
plt.scatter(X_pca[:, 0], X_pca[:, 1], c= clustering, cmap=matplotlib.colors.ListedColormap(colors))
for label, x, y in zip(labels, X_pca[:, 0], X_pca[:, 1]):
 plt.annotate(label, xy=(x, y), xytext=(-0.2, 0.2), textcoords='offset points')
plt.title("kmeans 3 cluster")
plt.show()

########### Aglomerative  #############

aglo_ward = AgglomerativeClustering(n_clusters = 3,linkage='ward').fit(X)
clustering = aglo_ward.labels_
Score["algo_ward"] = silhouette_score(X, clustering,metric='euclidean')

colors = ['red','yellow','blue','pink']
plt.scatter(X_pca[:, 0], X_pca[:, 1], c= clustering, cmap=matplotlib.colors.ListedColormap(colors))
for label, x, y in zip(labels, X_pca[:, 0], X_pca[:, 1]):
 plt.annotate(label, xy=(x, y), xytext=(-0.2, 0.2), textcoords='offset points')
plt.title("ward linkage")
plt.show()

aglo_average = AgglomerativeClustering(n_clusters = 3,linkage='average').fit(X)
clustering = aglo_average.labels_
Score["algo_average"] = silhouette_score(X, clustering,metric='euclidean')


colors = ['red','yellow','blue','pink']
plt.scatter(X_pca[:, 0], X_pca[:, 1], c= clustering, cmap=matplotlib.colors.ListedColormap(colors))
for label, x, y in zip(labels, X_pca[:, 0], X_pca[:, 1]):
 plt.annotate(label, xy=(x, y), xytext=(-0.2, 0.2), textcoords='offset points')
plt.title("average linkage")
plt.show()



########### silhouette ###############


for i in np.arange(2, 6):
 clustering = KMeans(n_clusters=i).fit_predict(X)
 print(silhouette_score(X, clustering,metric='euclidean'))
 print()
 
print("le nombre de de cluster pour avoir le score silhouette le plus élevé est de 2 cluster")
 
kmeans = KMeans(n_clusters=3, init='k-means++').fit_predict(X)

score = silhouette_score(X, kmeans)


########### réponse question 4 ################

max_key = max(Score, key=Score.get)

print("le clustering qui obtient le meilleur score silhouette pour 3 cluster est : ",max_key)

'''
On remarque que comparer  au clustering kmeans ou ward l'average ne vas pas classer les individus
centraux du graph en un 3 eme cluster mais plutot les incorporer au cluster contenant les élements les plus
a gauche du graph.

on arrive donc a un graph final avec 2 cluster principaux un a gauche et un droite du graph ainsi qu'un
3 eme cluster contenant qu'un seul élément situer en haut du graph. On pourais donc si on cherchait a
nommer les cluster nommer celui ci 


'''



