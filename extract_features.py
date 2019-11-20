
# coding: utf-8

# In[1]:


from imutils import paths
import random
import cv2
from AMOL.OBJECT_DETECTION import HELPERS
from AMOL.DESCRIPTOR.hog import HOG
from scipy import io
import progressbar
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from AMOL.UTILS import dataset
import h5py


# In[10]:


import argparse
AP = argparse.ArgumentParser()
AP.add_argument("-FN","--FOLDER_NAME",required=True)
AP.add_argument("-W","--WIDTH",required=True)
AP.add_argument("-H","--HEIGHT",required=True)
ARGS = vars(AP.parse_args())


# In[2]:



hog = HOG(orientations=9, pixelsPerCell=(4,4),
	cellsPerBlock=(2,2), normalize=True)

#print((ARGS['WIDTH'],ARGS['HEIGHT']))
# In[3]:


TRNPATH = list(paths.list_images("101_ObjectCategories"+"/"+ARGS['FOLDER_NAME']))
TRNPATH = random.sample(TRNPATH, int(len(TRNPATH) * 0.5))
DATA = []
LABELS=[]


# In[4]:


import imutils
import cv2
 
def CROP(image, bb,dstSize):
    padding=10
    #dstSize=(ARGS['WIDTH'],ARGS['HEIGHT'])
    (y, h, x, w) = bb
    (x, y) = (max(x - padding, 0), max(y - padding, 0))
    roi = image[y:h + padding, x:w + padding]
    roi = cv2.resize(roi, dstSize, interpolation=cv2.INTER_AREA)
    return roi


# In[5]:


widgets = ["Extracting: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(TRNPATH), widgets=widgets).start()


# In[6]:


for (i,TP) in enumerate(TRNPATH):
    image = cv2.imread(TP)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    IM_ID =TP[TP.rfind("_")+1:].replace(".jpg","")
    P = "Annotations/{}/annotation_{}.mat".format(ARGS['FOLDER_NAME'],IM_ID)
    BB = io.loadmat(P)["box_coord"][0]
    roi = CROP(image,BB,(int(ARGS['WIDTH']),int(ARGS['HEIGHT'])))
    features = hog.describe(roi)
    DATA.append(features)
    LABELS.append(1)
    pbar.update(i)
    


# In[7]:


pbar.finish()
dstPaths = list(paths.list_images('sceneclass13'))
pbar = progressbar.ProgressBar(maxval=500, widgets=widgets).start()
print("[INFO] describing distraction ROIs...")
 
# loop over the desired number of distraction images
for i in np.arange(0,500):
	# randomly select a distraction image, load it, convert it to grayscale, and
	# then extract random patches from the image
	image = cv2.imread(random.choice(dstPaths))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	patches = extract_patches_2d(image,(int(ARGS['WIDTH']),int(ARGS['HEIGHT'])),max_patches=10)
 
	# loop over the patches
	for patch in patches:
		# extract features from the patch, then update the data and label list
		features = hog.describe(patch)
		DATA.append(features)
		LABELS.append(-1)
 
	# update the progress bar
	pbar.update(i)


# In[8]:


pbar.finish()


# In[9]:



A = np.c_[LABELS,DATA]

from sklearn.svm import SVC
import pickle
from AMOL.OBJECT_DETECTION import HELPERS
import h5py
#db = h5py.File("output/cars/car3_features.hdf5", "r")
(labels, data) = (A[:,0],A[:,1:])
#db.close()
print("[INFO] training classifier...")
model = SVC(kernel="linear", C=0.01, probability=True, random_state=42)
model.fit(data, labels)
 
# dump the classifier to file
print("[INFO] dumping classifier...")
f = open("output/cars/"+ARGS['FOLDER_NAME'], "wb")
f.write(pickle.dumps(model))
f.close()

