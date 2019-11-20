
# coding: utf-8

# In[2]:


import cv2
import argparse


# In[1]:


from AMOL.OBJECT_DETECTION.HELPERS import PYRAMID


# In[3]:


AP = argparse.ArgumentParser()


# In[ ]:


AP.add_argument("-i","--image",required=True)
#AP.add_argument("-s","--scale")


# In[ ]:


ARGS = vars(AP.parse_args())


# In[19]:


#ARGS={'image':'ROB.jpg'}


# In[20]:


'''
def PYRAMID(IMAGE):
    import imutils
    scale =1.5
    yield IMAGE
    while True:
        WIDTH = int(IMAGE.shape[1]/scale)
        IMAGE = imutils.resize(IMAGE,width=WIDTH)
        if IMAGE.shape[0] <=30 or IMAGE.shape[1]<=30:
            break
        yield IMAGE
'''


# In[22]:


I =cv2.imread(ARGS["image"])
for i,layer in enumerate(PYRAMID(I)):
    cv2.imshow("layer {}".format(i+1),layer)
    cv2.waitKey(0)

