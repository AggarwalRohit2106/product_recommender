import numpy as np
import os
import pickle
import tensorflow
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
from tqdm import tqdm

# import resnet model
model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#extract 2048 feature for each 44k images

def features_extractor(img_path,model):                                   # return 2048 feature for each image
    img = image.load_img(img_path,target_size=(224,224))                  # load image
    array_img = image.img_to_array(img)                                   # converting into array
    expanded_img = np.expand_dims(array_img, axis=0)                      # expnad its dimension into 4 as resnet deals in bacthes
    preprocessed_img = preprocess_input(expanded_img)                     #process image into imagenet images specs
    result = model.predict(preprocessed_img).flatten()                    #feature extrction
    final_result = result / norm(result)                                  # normalise vector in 0 or 1

    return final_result

image_File_Name = []                                                      # list that store all images file name

for file in os.listdir('images'):
    image_File_Name.append(os.path.join('images',file))                   # storing filenames in list

feature_vector = []                                                       # 2D list that store 2048 feature for each 44k images(44K x 2048)

for file in tqdm(image_File_Name):
    feature_vector.append(features_extractor(file,model))                 #for each file in list extracitong features

pickle.dump(feature_vector,open('feature_vector.pkl','wb'))
pickle.dump(image_File_Name,open('filenames.pkl','wb'))