# for loading/processing the images  
import sklearn
import pandas as pd
#import matplotlib.pyplot
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 


# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# clustering and dimension reduction


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle

path = r"CHANGE TO DATASET LOCATION"
# change the working directory to the path where the images are located
os.chdir((r"C:\Users\rapha\Downloads\archive (1)\flower_images\flower_images") ) # tentei inserir "C:\Users\rapha\Downloads\archive (1)\flower_images\flower_images" porem nao funcionou 

# this list holds all the image filename
flowers = []

# creates a ScandirIterator aliased as files
with os.scandir((r"C:\Users\rapha\Downloads\archive (1)\flower_images\flower_images") ) as files:
  # loops through each file in the directory
    for file in files:
        if file.name.endswith('.png'):
          # adds only the image files to the flowers list
            flowers.append(file.name)
            
            
            
model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

def extract_features(file, model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features
   
data = {}
p = r"CHANGE TO A LOCATION TO SAVE FEATURE VECTORS"

# lop through each image in the dataset
for flower in flowers:
    # try to extract the features and update the dictionary
    try:
        feat = extract_features(flower,model)
        data[flower] = feat
    # if something fails, save the extracted features as a pickle file (optional)
    except:
        with open(p,'wb') as file:
            pickle.dump(data,file)
          
 
# get a list of the filenames
filenames = np.array(list(data.keys()))

# get a list of just the features
feat = np.array(list(data.values()))

# reshape so that there are 210 samples of 4096 vectors
feat = feat.reshape(-1,4096)

# get the unique labels (from the flower_labels.csv)
df = pd.read_csv('flower_labels.csv')
label = df['label'].tolist()
unique_labels = list(set(label))

# reduce the amount of dimensions in the feature vector
pca = PCA(n_components=100, random_state=22)
pca.fit(feat)
x = pca.transform(feat)

# cluster feature vectors
kmeans = KMeans(init="random", n_clusters=3, n_init=10, max_iter=300, random_state=42 )
kmeans.fit(x)

# holds the cluster id and the images { id: [images] }
groups = {}
for file, cluster in zip(filenames,kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)

# function that lets you view a cluster (based on identifier)        


def view_cluster(cluster):
    plt.figure(figsize = (25,25));
    # gets the list of filenames for a cluster
    files = groups[cluster]
    # only allow up to 30 images to be shown at a time
    if len(files) > 30:
        print(f"Clipping cluster size from {len(files)} to 30")
        files = files[:29]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(10,10,index+1);
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')
        
   
# this is just incase you want to see which value for k might be the best 
sse = []
list_k = list(range(3, 50))



for k in list_k:
    km = KMeans(init="random", n_clusters=3, n_init=10, max_iter=300, random_state=42 )
    km.fit(x)
    
    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse)
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance');

plt.show()

# Fixing random state for reproducibility
np.random.seed(19680801)

# Compute areas and colors
N = 150
r = 2 * np.random.rand(N)
theta = 2 * np.pi * np.random.rand(N)
area = 200 * r**2
colors = theta

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75) 

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)

ax.set_thetamin(45)
ax.set_thetamax(135)

plt.show()


# creating Kmeans object using  KMeans()
kmean = KMeans(n_clusters = 3, random_state=1)
# Fit on data
kmean.fit(X)
KMeans(algorithm='auto', 
       copy_x=True, 
       init='k-means++', # selects initial cluster centers
       max_iter=300,
       n_clusters=3, 
       n_init=10, 
       n_jobs=None, 
       precompute_distances='auto',
       random_state=1, 
       tol=0.0001, # min. tolerance for distance between clusters
       verbose=0)

# instantiate a variable for the centers
centers = kmean.cluster_centers_
# print the cluster centers
print(centers)