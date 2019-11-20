
# coding: utf-8

# In[64]:


from scipy import io
import numpy as np
import glob
import argparse


# In[66]:


AP = argparse.ArgumentParser()
AP.add_argument("-FN","--FOLDER_NAME",required=True)
ARGS =  vars(AP.parse_args())

# In[61]:




# In[62]:


WIDTH = []
HEIGHT = []
for P in glob.glob("Annotations"+"/"+ARGS['FOLDER_NAME']+"/*"):
    I = io.loadmat(P)
    (x1,x2,y1,y2) = I['box_coord'][0]
    WIDTH.append(y2-y1)
    HEIGHT.append(x2-x1)


# In[63]:

W= int(np.mean(WIDTH)/2)
H= int(np.mean(HEIGHT)/2)
while True:
    if int(W%4)==0:
        break
    else:
        W=W+1

while True:
    if int(H%4)==0:
        break
    else:
        H=H+1
    


print("AVERAGE WIDTH: {}".format(W))
print("AVERAGE HEIGHT: {}".format(H))
print("ASPECT RATIO: {}".format(np.mean(WIDTH)/np.mean(HEIGHT)))
