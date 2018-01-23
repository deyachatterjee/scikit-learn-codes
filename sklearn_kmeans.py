import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics
from sklearn import cluster
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
#sklearn has built in datasets like iris and digits.

#Load dataset
digits= datasets.load_digits()

"""
if not using built in data but some public dataset like the following
digits=pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra",header=None)
"""


#Visualize the data
# Figure size (width, height) in inches. blank figure
fig = plt.figure(figsize=(6, 6))

# layout adjustments for the subplots 
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# For each of the 64 images in 8x8 grid
for i in range(64):
    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    # Display an image at the i-th position
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    #The target labels are printed at coordinates (0,7) of each subplot, which means that they will appear in the bottom-left of each of the subplots.
    ax.text(0, 7, str(digits.target[i]))
plt.show()


#Since the image contains 64 features, high dimensionality might be a problem. We use sklearn's PCA function to visualise data in lower dimension.

randomized_pca=RandomizedPCA(n_components=2)   # you have two-dimensional data to plot
pca=PCA(n_components=2)

reduced_data_rpca=randomized_pca.fit_transform(digits.data)
reduced_data_pca=pca.fit_transform(digits.data)
print(reduced_data_pca.shape)
print(reduced_data_rpca.shape)
print(reduced_data_pca)
print(reduced_data_rpca)


#Now we build a scatterplot to visualize the data, which is lower dimensioned after PCA

#data points can be colored according to the labels
colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
for i in range(len(colors)):
    
	"""
	take the first or the second column of reduced_data_rpca, and you select only those data points for which the label equals the index that you’re considering. That means that in the first run, you’ll consider the data points with label 0, then label 1, … and so on.
	"""
	x = reduced_data_rpca[:, 0][digits.target == i]
    y = reduced_data_rpca[:, 1][digits.target == i]
    plt.scatter(x, y, c=colors[i])
	
plt.legend(digits.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title("PCA Scatter Plot")
plt.show()


#data preprocessing
data=scale(digits.data) #By scaling the data, we shift the distribution of each attribute to have a mean of zero and a standard deviation of one (unit variance).


#train and test sets
X_train,X_test,y_train,y_test,img_train,img_test=train_test_split(data,digits.target,digits.images,test_size=0.25,random_state=42)

n_samples,n_features= X_train.shape
n_digits=len(np.unique(y_train))
#training set 1347, test set 450


#use kmeans to cluster and classify

clf=cluster.Kmeans(init='k-means++',n_clusters=10,random_state=42)
clf.fit(X_train)



#visualise cluster centroids

fig = plt.figure(figsize=(8, 3))
fig.subtitle('Cluster Center Images', fontsize=14, fontweight='bold')

# For all labels (0-9)
for i in range(10):
    # Initialize subplots in a grid of 2X5, at i+1th position
    ax = fig.add_subplot(2, 5, 1 + i)
    # Display images
    ax.imshow(clf.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
    # Don't show the axes
    plt.axis('off')
plt.show()



#predicting

y_pred=clf.predict(X_test)
print (y_pred[:100])
print (y_test[:100])
clf.cluster_centers_.shape       #10 clusters with 64 features each



#more visualisation
#instead of PCA we use Isomap, as it is nonlinear dimensionality reduction method.
from sklearn.manifold import Isomap 
# Create an isomap and fit the `digits` data to it
X_iso = Isomap(n_neighbors=10).fit_transform(X_train)
# Compute cluster centers and predict cluster index for each sample
clusters = clf.fit_predict(X_train)
# Create a plot with subplots in a grid of 1X2
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# Adjust layout
fig.suptitle('Predicted Versus Training Labels', fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.85)

# Add scatterplots to the subplots 
ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=clusters)
ax[0].set_title('Predicted Training Labels')
ax[1].scatter(X_iso[:, 0], X_iso[:, 1], c=y_train)
ax[1].set_title('Actual Training Labels')
plt.show()



#what would happen if we used PCA in place of IsoMap?
from sklearn.decomposition import PCA
# Model and fit the `digits` data to the PCA model
X_pca = PCA(n_components=2).fit_transform(X_train)
# Compute cluster centers and predict cluster index for each sample
clusters = clf.fit_predict(X_train)
# Create a plot with subplots in a grid of 1X2
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# Adjust layout
fig.suptitle('Predicted Versus Training Labels', fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.85)
# Add scatterplots to the subplots 
ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
ax[0].set_title('Predicted Training Labels')
ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_train)
ax[1].set_title('Actual Training Labels')
plt.show()



#evaluate the model's performance
print (metrics.confusion_matrix(y_test,y_pred))


#applying various other cluster quality metrics...
from sklearn.metrics import homogeneity_score as h
from sklearn.metrics import completeness_score as c
from sklearn.metrics import v_measure_score as v
from sklearn.metrics import adjusted_rand_score as ar
from sklearn.metrics import adjusted_mutual_info_score as am
from sklearn.metrics import silhouette_score as s

print ('%9s' %'inertia	homo	comp	vmeas	ari	  ami	silh')
print ('%i	%.3f	%.3f	%.3f	%.3f	%.3f'%(clf.inertia_,h(y_test,y_pred),c(y_test,y_pred),v(y_test,y_pred),ar(y_test,y_pred),am(y_test,y_pred),s(y_test,y_pred,metric="euclidean")))
