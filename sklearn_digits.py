import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.cross_validation import train_test_split
#sklearn has built in datasets like iris and digits.

#Load dataset
digits= datasets.load_digits()

#print(digits.target)          #digits.target gives the number corresponding to each digit image.
#print(digits.images[0])

#it gives each original sample i.e. an image of shape (8,8).data is always a 2D array which has a shape (n_samples, n_features)

print(len(digits.data))      #length is 1796 examples 
 

#train and test sets
X_train,X_test,y_train,y_test,img_train,img_test=train_test_split(digits.data,digits.target,digits.images,test_size=0.25,random_state=42)

#Learning using SVM classifier
clf = svm.SVC(gamma=0.001, C=100,kernel="linear")

#fit data to model
clf.fit(X_train,y_train)

#note: we can also use gridsearch to adjust parameters. 

#view accuracy score
clf.score(X_test,y_test)

print (clf.predict(X_test))
print (y_test)

#visualize the images and their predicted labels

# Assign the predicted values to `predicted`
predicted = svc_model.predict(X_test)
images_and_predictions = list(zip(images_test, predicted))
# For the first 4 elements in `images_and_predictions`
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    # Initialize subplots in a grid of 1 by 4 at positions i+1
    plt.subplot(1, 4, index + 1)
    # Don't show axes
    plt.axis('off')
    # Display images in all subplots in the grid
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Predicted: ' + str(prediction))
plt.show()

#evaluate performance
print (metrics.classification_report(y_test,predicted))
print (metrics.confusion_matrix(y_test,predicted))

#visualize the predicted and the actual labels with Isomap():

from sklearn.manifold import Isomap

# Create an isomap and fit the `digits` data to it
X_iso = Isomap(n_neighbors=10).fit_transform(X_train)

# Compute cluster centers and predict cluster index for each sample
predicted = svc_model.predict(X_train)

# Create a plot with subplots in a grid of 1X2
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# Adjust the layout
fig.subplots_adjust(top=0.85)

# Add scatterplots to the subplots 
ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=predicted)
ax[0].set_title('Predicted labels')
ax[1].scatter(X_iso[:, 0], X_iso[:, 1], c=y_train)
ax[1].set_title('Actual Labels')

fig.suptitle('Predicted versus actual labels', fontsize=14, fontweight='bold')
plt.show()

#svm performs better than kmeans