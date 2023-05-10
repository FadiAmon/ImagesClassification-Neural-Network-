# import necessary libraries
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

# load the training and validation data from CSV files
df = pd.read_csv('training_ex3_dl2022b.csv')
validate_df = pd.read_csv('validation_ex3_dl2022b.csv')

# create a list of tuples for each image, including its ID, URL, and label
result_train = zip(df['id'].tolist(), df['image'].tolist(),df['label'].tolist())
result_validate = zip(validate_df['id'].tolist(), validate_df['image'].tolist(),validate_df['label'].tolist())

# create a dictionary of training images where the keys are the image IDs
# this dictionary will be used later to train the model
images_ids=[]
for images in os.listdir("training_images"):
    images_ids.append(images[:-4])
train_dic={}
count=0
for ID,image_url,label in result_train:
    if (str(ID) in images_ids):
        count+=1
        train_dic[ID]=(image_url,label)

# create a dictionary of validation images where the keys are the image IDs
# this dictionary will be used later to evaluate the model
images_ids=[]
for images in os.listdir("validation_images"):
    images_ids.append(images[:-4])
val_dic={}
count=0
for ID,image_url,label in result_validate:
    if (str(ID) in images_ids):
        count+=1
        val_dic[ID]=(image_url,label)

# print the number of training and validation images
print(len(train_dic))
print(len(val_dic))

def download_image(result, File):
    """
    Downloads images from the given list of URLs and stores them in a specified directory.

    Args:
    - result: A list of tuples containing the image IDs, URLs, and corresponding labels.
    - File: The directory where the images will be stored.

    Returns:
    - Id_Image_Label_Dict: A dictionary containing the image IDs as keys and their corresponding URLs and labels as values.

    """
    count = 1
    Id_Image_Label_Dict = {}

    # Loop over each tuple in the input list and download the corresponding image
    for ID, image_url, label in result:
        count = count + 1

        # Attempt to download the image using the requests module
        try:
            req = requests.get(image_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=120)

            # Check if the site is returning a proper response
            if (req.status_code != 200):
                print(count)
                print('site is not proper')
                continue
        # Handle various types of exceptions that can occur during the request
        except requests.exceptions.ConnectionError as e:
            print(count) #num of problematic link
            print(e)
            continue
        except KeyboardInterrupt as e:
            raise e
        except:
            print(count) #num of problematic link
            print("caught a timeout")
            continue

        # Write the image content to a file with the corresponding ID as the filename
        file = open(rf"{File}" + "\\" + str(ID) + ".jpg", "wb")
        try:
            file.write(req.content)
            Id_Image_Label_Dict[ID] = (image_url, label)
            file.close()
        except Exception as e:
            file.close()
            raise e

    return Id_Image_Label_Dict

# remove the comments if you want to download the train and validation images using the download function!
# make sure to comment the code above where it says you can get dict without using the download function!
#train_dic=download_image(result_train,"training_images")
#val_dic=download_image(result_validate,"validation_images")

# create empty lists to store the training and validation images and their corresponding labels
train_images=[]
train_labels=[]
test_images=[]
test_labels=[]




def modify_files(folder_dir, image_dic):
    """
    A function that modifies images in the given folder directory and saves them. It also creates two lists of 
    training and testing images and their corresponding labels. 
    
    Args:
    folder_dir (str): The directory where the images are stored. It can either be "training_images" or 
    "validation_images".
    image_dic (dict): A dictionary containing the ID, image name and label of each image in the folder directory.
    
    Returns:
    fit_dic (dict): A dictionary containing the ID, image name and label of each image in the folder directory.
    """
    
    # A dictionary to store the ID, image name and label of each image
    fit_dic={}
    
    # Enable truncated images to avoid errors when loading images
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    # Iterate through all images in the folder directory
    for images in os.listdir(folder_dir):
        # check if the image ends with png
        if (images.endswith(".jpg")):
            try:
                # Open the image
                image = Image.open(folder_dir+"\\"+images)
                
            except Exception as e:
                # If there is an error loading the image, delete the image and remove its details from the dictionary
                os.remove(folder_dir+"\\"+images)
                del image_dic[int(images[:-4])]
                print(f"Image {images[:-4]} deleted")
                continue
            
            # Convert image mode to RGB if it is not already in RGB
            if (image.mode!="RGB"):
                image = image.convert('RGB')
            
            # Resize the image to 128x128
            image = image.resize((128, 128))
            
            # Save the modified image
            image.save(folder_dir+"\\"+images)
            
            # If the folder directory is "training_images", add the image to the training list and its label
            # to the training label list. Otherwise, add the image to the testing list and its label to the testing 
            # label list
            if (folder_dir=="training_images"):
                fit_dic[int(images[:-4])]=image_dic[int(images[:-4])]
                train_images.append((np.array(image).astype('float32'))/255)
                train_labels.append(image_dic[int(images[:-4])][1])
            else:
                fit_dic[int(images[:-4])]=image_dic[int(images[:-4])]
                test_images.append((np.array(image).astype('float32'))/255)
                test_labels.append(image_dic[int(images[:-4])][1])
    
    # Return the dictionary containing the ID, image name and label of each image
    return fit_dic


# Call the modify_files() function to modify the validation images
VAL_Id_Image_Label_Dict = modify_files("validation_images", val_dic)

# Print the length of the testing images, testing labels and dictionary containing the ID, image name and label of
# each testing image
print(len(test_images),len(test_labels),len(VAL_Id_Image_Label_Dict))

# Call the modify_files() function to modify the training images
TRAIN_Id_Image_Label_Dict = modify_files("training_images", train_dic)

# Print the length of the training images, training labels and dictionary containing the ID, image name and label of
# each training image
print(len(train_images),len(train_labels),len(TRAIN_Id_Image_Label_Dict))


# A dictionary that maps each fruit name to a unique class index.
class_dict = {'bergamot': 0, 'pomelo': 1, 'yuzu': 2, 'lemon': 3, 'mandarin': 4, 'orange': 5, 'tangerine': 6}

# Convert test_labels to class indices using the dictionary.
for index in range(0, len(test_labels)):
    test_labels[index] = class_dict[test_labels[index]]

# Convert train_labels to class indices using the dictionary.
for index in range(0, len(train_labels)):
    train_labels[index] = class_dict[train_labels[index]]

# Print the length of the test data and labels.
print(len(test_images), len(test_labels))

# Print the length of the training data and labels.
print(len(train_images), len(train_labels))

# Convert the lists of labels and images to numpy arrays.
test_labels = np.asarray(test_labels)
train_labels = np.asarray(train_labels)
train_images = np.asarray(train_images)
test_images = np.asarray(test_images)

# Create a sequential model.
model = models.Sequential()

# Add a convolutional layer with 64 filters, a 2x2 kernel size, and ReLU activation function.
# Input shape is 128x128x3.
model.add(layers.Conv2D(filters=64, kernel_size=(2,2), padding="same", activation="relu", input_shape=(128, 128, 3)))

# Add a max pooling layer with a pool size of 2x2.
model.add(layers.MaxPooling2D(pool_size=(2,2)))

# Add another convolutional layer with 64 filters, a 2x2 kernel size, and ReLU activation function.
model.add(layers.Conv2D(filters=64, kernel_size=(2,2), padding="same", activation="relu"))

# Add another max pooling layer with a pool size of 2x2.
model.add(layers.MaxPooling2D(pool_size=(2,2)))

# Add another convolutional layer with 64 filters, a 2x2 kernel size, and ReLU activation function.
model.add(layers.Conv2D(filters=64, kernel_size=(2,2), padding="same", activation="relu"))

# Add another max pooling layer with a pool size of 2x2.
model.add(layers.MaxPooling2D(pool_size=(2,2)))

# Add another convolutional layer with 64 filters, a 2x2 kernel size, and ReLU activation function.
model.add(layers.Conv2D(filters=64, kernel_size=(2,2), padding="same", activation="relu"))

# Add another max pooling layer with a pool size of 2x2.
model.add(layers.MaxPooling2D(pool_size=(2,2)))

# Add another convolutional layer with 64 filters, a 2x2 kernel size, and ReLU activation function.
model.add(layers.Conv2D(filters=64, kernel_size=(2,2), padding="same", activation="relu"))

# Add another max pooling layer with a pool size of 2x2.
model.add(layers.MaxPooling2D(pool_size=(2,2)))

# Flatten the output of the previous layer to a 1D array.
model.add(layers.Flatten())

# Add a fully connected layer with 500 units and ReLU activation function.
model.add(layers.Dense(500, activation="relu"))

# Add a fully connected output layer with 7 units and softmax activation function.
model.add(layers.Dense(7, activation="softmax"))

# Compile the Keras model with Adam optimizer, Sparse Categorical Crossentropy loss, and Accuracy metrics
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

# Set up early stopping to monitor validation accuracy and stop training if it doesn't improve after 4 epochs
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='min', verbose=4, patience=4)

# Train the model on the training data with 50 epochs, a batch size of 5, and the validation data, using early stopping
history = model.fit(train_images, train_labels, epochs=50, batch_size=5, validation_data=(test_images, test_labels), callbacks=[early_stopping])

# Read in the CSV file with the test data and create a list of tuples with the image IDs and URLs
test_df = pd.read_csv('test_ex3_dl2022b.csv')
result_test = zip(test_df['id'].tolist(), test_df['image'].tolist())

def download_test(result, File):
    """
    Downloads the test images from a given list of image URLs and saves them to a specified directory.
    
    Args:
    - result: A list of tuples, where each tuple contains an image ID and its URL.
    - File: A string representing the path to the directory where the images will be saved.
    
    Returns:
    - A dictionary where the keys are image IDs and the values are their respective URLs.
    """
    count = 1 # Initialize a counter for the loop
    Id_Image_Dict = {} # Initialize an empty dictionary for storing the image IDs and URLs
    for ID, image_url in result: # Loop through each image ID and URL in the given list
        count += 1 # Increment the counter
        try:
            # Use requests library to get the image from its URL
            req = requests.get(image_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=120)
            # Check the HTTP status code to make sure the site is proper
            if (req.status_code != 200):
                print(count)
                print('site is not proper')
                continue # If the status code is not 200 (OK), skip to the next image
        except requests.exceptions.ConnectionError as e:
            print(count)  # num of problematic link
            print(e)
            continue # If there is a connection error, skip to the next image
        except KeyboardInterrupt as e:
            raise e # If the user interrupts the program, raise an exception to stop the function
        except:
            print(count)  # num of problematic link
            print("caught a timeout")
            continue # If there is a timeout, skip to the next image
        # Open a file with write binary mode and write the content of the image to it
        file = open(rf"{File}" + "\\" + str(ID)+".jpg", "wb")
        try:
            file.write(req.content)
            Id_Image_Dict[ID] = image_url # Add the ID and URL to the dictionary
            file.close() # Close the file
        except Exception as e:
            file.close() # Close the file if an exception is raised
            raise e # Raise the exception to stop the function
    return Id_Image_Dict # Return the dictionary of image IDs and URLs

# Download the test images using the download_test function (uncomment the line below to use this function)
# test_dic = download_test(result_test, "testing_images")

# Create a list of IDs for the images that were already downloaded to the folder
images_ids = []
for images in os.listdir("testing_images"):
    images_ids.append(images[:-4])

count = 0
test_dic = {}
# Create a dictionary of image IDs and URLs for the images that are already downloaded
for ID, image_url in result_test:
    if (str(ID) in images_ids):
        count += 1
        test_dic[ID] = image_url

# Print the length of the dictionary of test images
print(len(test_dic))

# Create an empty list to hold the preprocessed test images
test_images = []


def modify_files_test(folder_dir: str, image_dic: dict) -> dict:
    """
    This function modifies test images by resizing them to 128x128 pixels, converting them to RGB format
    and saving the modified images back to the same folder. It also returns a dictionary of image IDs and
    URLs for the modified images that could be used for model testing.

    :param folder_dir: str, the path to the folder containing the test images
    :param image_dic: dict, a dictionary of image IDs and URLs for the original test images
    :return: dict, a dictionary of image IDs and URLs for the modified test images that could be used for model testing
    """
    fit_dic = {} # Create an empty dictionary to store the fit images
    ImageFile.LOAD_TRUNCATED_IMAGES = True # Set an option to load truncated images
    for images in os.listdir(folder_dir): # Loop over the images in the test folder
        # Check if the image file ends with ".jpg"
        if (images.endswith(".jpg")):
            try:
                image = Image.open(folder_dir + "\\" + images) # Open the image file
            except Exception as e:
                # If the image cannot be opened, delete it and remove it from the image dictionary
                os.remove(folder_dir + "\\" + images)
                del image_dic[int(images[:-4])]
                print(f"Image {images[:-4]} deleted")
                continue
            if (image.mode != "RGB"):
                image = image.convert('RGB') # Convert the image to RGB format if it is not already
            image = image.resize((128, 128)) # Resize the image to (128, 128)
            image.save(folder_dir + "\\" + images) # Save the modified image file
            fit_dic[int(images[:-4])] = image_dic[int(images[:-4])] # Add the fit image to the fit dictionary
            test_images.append((np.array(image).astype('float32')) / 255) # Add the modified image to the test images list

    return fit_dic # Return the fit dictionary

# Call the modify_files_test function to modify the test images
TEST_Id_Image_Dict = modify_files_test("testing_images", test_dic)

# Print the number of images in the fit dictionary
print(len(TEST_Id_Image_Dict))

# Convert the test images list to a numpy array
test_images=np.asarray(test_images)
print(len(test_images))

# Use the trained model to predict the classes of the test images
predictions=model.predict(test_images)

# Convert the predicted classes to a list of labels
test_predictions=[]
for x in predictions:
    test_predictions.append(list(x).index(max(list(x))))

# Create a dictionary to map the predicted classes to their corresponding labels
class_dict={'bergamot': 0, 'pomelo': 1, 'yuzu': 2, 'lemon': 3, 'mandarin': 4, 'orange': 5, 'tangerine': 6}

# Convert the predicted classes to their corresponding labels
key_list = list(class_dict.keys())
val_list = list(class_dict.values())
for index in range(0,len(test_predictions)):
    position = val_list.index(test_predictions[index])
    test_predictions[index]=key_list[position]

# Assign the predicted labels to the fit dictionary
index=0
for ID in list(TEST_Id_Image_Dict.keys()):
    TEST_Id_Image_Dict[ID]=test_predictions[index]
    index+=1

# Read the sample submission csv file
df_submit = pd.read_csv('sample_submission_ex3_dl2022b.csv')

# Create a zip object to iterate over the ID and label columns of the sample submission csv file
id_label=zip(df_submit['id'].tolist(),df_submit['label'].tolist())

# Initialize counters for the total number of images and the number of images in the fit dictionary
all_count=0
dict_count=0

# Create empty lists to store the IDs and labels for the final submission csv file
all_ids=[]
all_labels=[]
# This loop iterates through each ID and label in id_label list
for ID, label in id_label:
    
    # Appends the ID to all_ids list
    all_ids.append(ID)
    
    # Checks if ID exists in the TEST_Id_Image_Dict dictionary
    if (ID in TEST_Id_Image_Dict.keys()):
        
        # If ID exists, it appends the predicted label to all_labels list 
        # using the ID as key to access the predicted label in TEST_Id_Image_Dict dictionary
        all_labels.append(TEST_Id_Image_Dict[int(ID)])
        
        # Increment the count of IDs with predicted labels 
        dict_count+=1
    
    else:
        # If ID doesn't exist, it appends 'lemon' as label to all_labels list
        all_labels.append('lemon')
    
    # Increment the count of all IDs in the id_label list
    all_count+=1 
    
# Sets the 'id' column in df_submit dataframe to all_ids list
df_submit['id']=all_ids

# Sets the 'label' column in df_submit dataframe to all_labels list
df_submit['label']=all_labels

# Writes the df_submit dataframe to a csv file named 'results_128.csv'
# with index set to False to exclude index column in the output file
df_submit.to_csv('results_128.csv',index=False)

