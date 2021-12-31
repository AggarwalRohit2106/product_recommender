import streamlit as st
import os
from PIL import Image
import numpy as np

import pickle
import tensorflow
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors

feature_vector= np.array(pickle.load(open('feature_vector.pkl','rb')))
files_Name= pickle.load(open('filenames.pkl','rb'))


model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


st.title("Myntra fashionable Recommender Engine")
st.caption("Just upload image, Get best recommendation")

# funtion that save uploaded filr in upload folder

def save_file(upload_file):
    try:
        with open(os.path.join("uploaded_image", upload_file.name), "wb") as f:
            f.write(upload_file.getbuffer())
        return 1

    except:
        return 0

# function that generates feature for uploaded image
def features_extractor(img_path,model):                                   # return 2048 feature for each image
    img = image.load_img(img_path,target_size=(224,224))                  # load image
    array_img = image.img_to_array(img)                                   # converting into array
    expanded_img = np.expand_dims(array_img, axis=0)                      # expnad its dimension into 4 as resnet deals in bacthes
    preprocessed_img = preprocess_input(expanded_img)                     #process image into imagenet images specs
    result = model.predict(preprocessed_img).flatten()                    #feature extrction
    final_result = result / norm(result)                                  # normalise vector in 0 or 1

    return final_result

# funciton that return indices of similar item
def recommender(feature,feature_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm="brute", metric="euclidean")
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([feature])
    return indices

upload_file = st.file_uploader("Drop your image")
if upload_file is not None:
    save_file(upload_file)
    disp_img= Image.open(upload_file)
    st.image(disp_img,width=300)
    features = features_extractor(os.path.join("uploaded_image",upload_file.name),model)  # feature contain 2048 feature of uploaded image
    indices= recommender(features,feature_vector)
    st.caption("You might also like :)")
    col1, col2, col3,col4 ,col5  = st.columns(5)
    with col1:
        st.image(files_Name[indices[0][0]])
    with col2:
        st.image(files_Name[indices[0][1]])
    with col3:
        st.image(files_Name[indices[0][2]])
    with col4:
        st.image(files_Name[indices[0][3]])
    with col5:
        st.image(files_Name[indices[0][4]])

else:
    print("error enocuntered")