#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import the modules
import os
from os import listdir
import numpy as np
from PIL import Image, ImageFile
import pandas as pd
import requests
import urllib.request
import ssl
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


# In[ ]:


df = pd.read_csv('training_ex3_dl2022b.csv')
validate_df = pd.read_csv('validation_ex3_dl2022b.csv')


# In[ ]:


result_train = zip(df['id'].tolist(), df['image'].tolist(),df['label'].tolist())
result_validate = zip(validate_df['id'].tolist(), validate_df['image'].tolist(),validate_df['label'].tolist())


# In[ ]:


#get dict without the download function
images_ids=[]
for images in os.listdir("training_images"):
    images_ids.append(images[:-4])
train_dic={}
count=0
#print(list(result_train))
for ID,image_url,label in result_train:
    if (str(ID) in images_ids):
        count+=1
        train_dic[ID]=(image_url,label)


# In[ ]:


#get dict without the download function
images_ids=[]
for images in os.listdir("validation_images"):
    images_ids.append(images[:-4])
val_dic={}
count=0
#print(list(result_train))
for ID,image_url,label in result_validate:
    if (str(ID) in images_ids):
        count+=1
        val_dic[ID]=(image_url,label)


# In[ ]:


print(len(train_dic))


# In[ ]:


print(len(val_dic))


# In[ ]:


def download_image(result,File):
    count=1
    Id_Image_Label_Dict={}
    for ID,image_url,label in result:
        count=count+1
        try:
            req = requests.get(image_url, headers={'User-Agent': 'Mozilla/5.0'},timeout=120)
            if (req.status_code!=200):
                print(count)
                print('site is not proper')
                continue
        except requests.exceptions.ConnectionError as e:
            print(count) #num of problamatic link
            print(e)
            continue
        except KeyboardInterrupt as e:
            raise e
        except:
            print(count) #num of problamatic link
            print ("caught a timeout")
            continue
        file = open(rf"{File}" + "\\" + str(ID)+".jpg", "wb")
        try:
            file.write(req.content)
            Id_Image_Label_Dict[ID]=(image_url,label)
            file.close()
        except Exception as e:
            file.close()
            raise e
    return Id_Image_Label_Dict


# In[ ]:


# remove the comments if you want to download the train and validation images using the download function!
# make sure to comment the code above where it says you can get dict without using the download function!
#train_dic=download_image(result_train,"training_images")
#val_dic=download_image(result_validate,"validation_images")


# In[ ]:


train_images=[]
train_labels=[]

test_images=[]
test_labels=[]


# In[ ]:


def modify_files(folder_dir, image_dic):
    fit_dic={}
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    for images in os.listdir(folder_dir):
        # check if the image ends with png
        if (images.endswith(".jpg")):
            try:
                image = Image.open(folder_dir+"\\"+images)
                
            except Exception as e:
                os.remove(folder_dir+"\\"+images)
                del image_dic[int(images[:-4])]
                print(f"Image {images[:-4]} deleted")
                continue
            if (image.mode!="RGB"):
                image = image.convert('RGB')                      
            image = image.resize((128, 128))
            image.save(folder_dir+"\\"+images)
            if (folder_dir=="training_images"):
                fit_dic[int(images[:-4])]=image_dic[int(images[:-4])]
                train_images.append((np.array(image).astype('float32'))/255)
                train_labels.append(image_dic[int(images[:-4])][1])
            else:
                fit_dic[int(images[:-4])]=image_dic[int(images[:-4])]
                test_images.append((np.array(image).astype('float32'))/255)
                test_labels.append(image_dic[int(images[:-4])][1])
                
            
    return fit_dic


# In[ ]:


VAL_Id_Image_Label_Dict = modify_files("validation_images", val_dic)


# In[ ]:


print(len(test_images),len(test_labels),len(VAL_Id_Image_Label_Dict))


# In[ ]:


TRAIN_Id_Image_Label_Dict = modify_files("training_images", train_dic)


# In[ ]:


print(len(train_images),len(train_labels),len(TRAIN_Id_Image_Label_Dict))


# In[ ]:


class_dict={'bergamot': 0, 'pomelo': 1, 'yuzu': 2, 'lemon': 3, 'mandarin': 4, 'orange': 5, 'tangerine': 6}
for index in range(0,len(test_labels)):
    test_labels[index]=class_dict[test_labels[index]]

for index in range(0,len(train_labels)):
    train_labels[index]=class_dict[train_labels[index]]


# In[ ]:


print(len(test_images),len(test_labels))


# In[ ]:


print(len(train_images),len(train_labels))


# In[ ]:


test_labels=np.asarray(test_labels)
train_labels=np.asarray(train_labels)
train_images=np.asarray(train_images)
test_images=np.asarray(test_images)


# In[ ]:


model = models.Sequential()

model.add(layers.Conv2D(filters=64,kernel_size=(2,2),padding="same",activation="relu",input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(filters=64,kernel_size=(2,2),padding="same",activation="relu",input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(filters=64,kernel_size=(2,2),padding="same",activation="relu",input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(filters=64,kernel_size=(2,2),padding="same",activation="relu",input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(filters=64,kernel_size=(2,2),padding="same",activation="relu",input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D(pool_size=(2,2)))


# In[ ]:


model.add(layers.Flatten())
model.add(layers.Dense(500,activation="relu"))
model.add(layers.Dense(7,activation="softmax"))


# In[ ]:


model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])


# In[ ]:


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='min', verbose=4, patience=4)
history=model.fit(train_images,train_labels, epochs = 50, batch_size=5, validation_data=(test_images, test_labels), callbacks=[early_stopping],)


# In[ ]:


test_df = pd.read_csv('test_ex3_dl2022b.csv')
result_test = zip(test_df['id'].tolist(), test_df['image'].tolist())


# In[ ]:


def download_test(result,File):
    count=1
    Id_Image_Dict={}
    for ID,image_url in result:
        count=count+1
        try:
            req = requests.get(image_url, headers={'User-Agent': 'Mozilla/5.0'},timeout=120)
            if (req.status_code!=200):
                print(count)
                print('site is not proper')
                continue
        except requests.exceptions.ConnectionError as e:
            print(count) #num of problamatic link
            print(e)
            continue
        except KeyboardInterrupt as e:
            raise e
        except:
            print(count) #num of problamatic link
            print ("caught a timeout")
            continue
        file = open(rf"{File}" + "\\" + str(ID)+".jpg", "wb")
        try:
            file.write(req.content)
            Id_Image_Dict[ID]=image_url
            file.close()
        except Exception as e:
            file.close()
            raise e
    return Id_Image_Dict


# In[ ]:


# remove the comments if you want to download the test images using the download function!
# make sure to comment the code at the block under this one, where it says you can get dict without using the download function!
#test_dic=download_test(result_test,"testing_images")


# In[ ]:


#get dict without the download function
images_ids=[]
for images in os.listdir("testing_images"):
    images_ids.append(images[:-4])
count=0
#print(list(result_train))
test_dic={}
for ID,image_url in result_test:
    if (str(ID) in images_ids):
        count+=1
        test_dic[ID]=image_url


# In[ ]:


print(len(test_dic))


# In[ ]:


test_images=[]


# In[ ]:


def modify_files_test(folder_dir, image_dic):
    fit_dic={}
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    for images in os.listdir(folder_dir):
        # check if the image ends with png
        if (images.endswith(".jpg")):
            try:
                image = Image.open(folder_dir+"\\"+images)
                
            except Exception as e:
                os.remove(folder_dir+"\\"+images)
                del image_dic[int(images[:-4])]
                print(f"Image {images[:-4]} deleted")
                continue
            if (image.mode!="RGB"):
                image = image.convert('RGB')                      
            image = image.resize((128, 128))
            image.save(folder_dir+"\\"+images)
            fit_dic[int(images[:-4])]=image_dic[int(images[:-4])]
            test_images.append((np.array(image).astype('float32'))/255)

    return fit_dic


# In[ ]:


TEST_Id_Image_Dict = modify_files_test("testing_images", test_dic)


# In[ ]:


print(len(TEST_Id_Image_Dict))


# In[ ]:


test_images=np.asarray(test_images)
print(len(test_images))


# In[ ]:


predictions=model.predict(test_images)


# In[ ]:


test_predictions=[]
for x in predictions:
    test_predictions.append(list(x).index(max(list(x))))


# In[ ]:


#class_dict={'bergamot': 0, 'pomelo': 1, 'yuzu': 2, 'lemon': 3, 'mandarin': 4, 'orange': 5, 'tangerine': 6}
key_list = list(class_dict.keys())
val_list = list(class_dict.values())
for index in range(0,len(test_predictions)):
    position = val_list.index(test_predictions[index])
    test_predictions[index]=key_list[position]


# In[ ]:


#fitting positions.
index=0
for ID in list(TEST_Id_Image_Dict.keys()):
    TEST_Id_Image_Dict[ID]=test_predictions[index]
    index+=1


# In[ ]:


df_submit = pd.read_csv('sample_submission_ex3_dl2022b.csv')


# In[ ]:


id_label=zip(df_submit['id'].tolist(),df_submit['label'].tolist())
all_count=0
dict_count=0
all_ids=[]
all_labels=[]
for ID, label in id_label:
    all_ids.append(ID)
    if (ID in TEST_Id_Image_Dict.keys()):
        all_labels.append(TEST_Id_Image_Dict[int(ID)])
        dict_count+=1
    else:
        all_labels.append('lemon')
    all_count+=1   


# In[ ]:


df_submit['id']=all_ids
df_submit['label']=all_labels


# In[ ]:


df_submit.to_csv('results_128.csv',index=False) #saving the predictions to csv.

