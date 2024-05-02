#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
img = cv2.imread(r"C:\Users\Adrian Calvin\OneDrive\Desktop\cp-vision\asl\train\asl_dataset\a\hand1_a_bot_seg_1_cropped.jpeg")
plt.imshow(img)




# In[5]:


cv2.imread(r"C:\Users\Adrian Calvin\OneDrive\Desktop\cp-vision\asl\train\asl_dataset\a\hand1_a_bot_seg_1_cropped.jpeg").shape



# In[11]:


train=ImageDataGenerator(rescale=1/255)
validation=ImageDataGenerator(rescale=1/255)


# In[12]:


train_dataset=train.flow_from_directory(r'C:\Users\Adrian Calvin\OneDrive\Desktop\cp-vision\asl\train\asl_dataset',
                                        target_size=(200,200),batch_size=40,class_mode='sparse')



# In[13]:


validation_dataset=train.flow_from_directory(r'C:\Users\Adrian Calvin\OneDrive\Desktop\cp-vision\asl\validation',
                                        target_size=(200,200),batch_size=40,class_mode='sparse')


# In[14]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy


# In[15]:


# Create the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')
])


# In[16]:


loss_function = SparseCategoricalCrossentropy()
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
model.save('trained_model.keras')


# In[17]:


batch_size = 40
num_epochs = 10

model.fit(train_dataset, validation_data=validation_dataset, batch_size=batch_size, epochs=num_epochs)


# In[18]:


from keras.preprocessing import image
j=0
target_element = 1
dir_path = r'C:\Users\Adrian Calvin\OneDrive\Desktop\cp-vision\asl\test'
for i in os.listdir(dir_path):
    img = image.load_img(dir_path + '//' + i, target_size=(200, 200))
    plt.imshow(img)
    plt.show()
    X= image.img_to_array(img)
    X= np.expand_dims(X,axis=0)
    images=np.vstack([X])
    val=model.predict(images)
    positions = np.where(val == target_element)
    # The result is a tuple of arrays containing the row and column indices
    row_indices, col_indices = positions
    print("Column Indices:", col_indices)
    print(val)
    if col_indices==0:
        op=cv2.imread(r'C:\Users\Adrian Calvin\OneDrive\Desktop\cp-vision\isl\test\a\2.png')
    elif col_indices==1:
        op=cv2.imread(r'C:\Users\Adrian Calvin\OneDrive\Desktop\cp-vision\isl\test\b\2.png')
    elif col_indices==2:
        op=cv2.imread(r'C:\Users\Adrian Calvin\OneDrive\Desktop\cp-vision\isl\test\c\2.png')
    elif col_indices==3:
        op=cv2.imread(r'C:\Users\Adrian Calvin\OneDrive\Desktop\cp-vision\isl\test\d\1.png')
    elif col_indices==4:
        op=cv2.imread(r'C:\Users\Adrian Calvin\OneDrive\Desktop\cp-vision\isl\test\e\2.png')
    else:
        print("error")
    output="ISL"
    cv2.imshow(output+"",op)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[ ]:





# In[40]:





# In[44]:


get_ipython().run_line_magic('pinfo2', 'model.save')


# In[ ]:




