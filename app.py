import streamlit as st
import tensorflow as tf
import cv2
import os
from PIL import Image, ImageOps
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
st.set_option('deprecation.showfileUploaderEncoding', False) # to avoid warnings while uploading files

# Here we will use st.cache so that we would load the model only once and store it in the cache memory which will avoid re-loading of model again and again.
@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('my_model_malaria_custom.hdf5')
  return model

# load and store the model
with st.spinner('Model is being loaded..'):
  model=load_model()

# Function for prediction
def import_and_predict(image_data, model):
#     size = (64,64)
#     image = Image.open(image_data)
#     image = image.resize((SIZE, SIZE))
#     image = np.asarray(image)
#     img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     img_reshape = img[np.newaxis,...]
    img = image.load_img(image_data, target_size=(64, 64))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)

    prediction = model.predict(x)
#     prediction = model.predict(image)
    return prediction
def main():
    st.title("Malaria Detection")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Malaria Detection using CNN </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    file = st.file_uploader("Please upload an image", type=["jpg", "png"])
    class_names=['parasitized','uninfected']
    result=""
    final_images=""
    with st.sidebar:
      with st.expander("Upload an image of one of these categories"):

#       st.header("Please upload an image from one of these categories")
        st.text("1. parasitized")
        st.text("2. uninfected")
      st.header("Malaria Detection using CNN")
      st.image("vgg16.jpg")

    if st.button("Predict"):
      if file is None:
        st.write("please upload an image")
      else:
        image = Image.open(file)

        predictions = import_and_predict(image,model)
        score = tf.nn.softmax(predictions[0])
        result= class_names[np.argmax(predictions[0])]
#         st.write('This is {} '.format(result))
        html_temp = f"""
                    <div style="background-color:tomato;padding:10px">
                    <h2 style="color:white;text-align:center;"> This is {result} </h2>
                    </div>
                     """
        st.markdown(html_temp,unsafe_allow_html=True)
        st.image(image, use_column_width=True)


        st.caption("The result is trained on similar images like: ")

        parasitized=[]
        uninfected=[]


        for folder_name in ['parasitized/','uninfected/']:

          #Path of the folder
          images_path = os.listdir(folder_name)

          for i, image_name in enumerate(images_path):
            if folder_name=='parasitized/':
                parasitized.append(folder_name+image_name)
            elif folder_name=='uninfected/':
                uninfected.append(folder_name+image_name)

        parasitized_list=[]
        uninfected_list=[]


        for i in parasitized:
          image = Image.open(i).resize((64, 64))
          parasitized_list.append(image)
        for i in uninfected:
          image = Image.open(i).resize((64, 64))
          uninfected_list.append(image)




        if result=='parasitized':
            final_images =parasitized_list

        elif result=='uninfected':
            final_images =uninfected_list


        n_rows = 1 + len(final_images) // int(4)
        rows = [st.container() for _ in range(n_rows)]
        cols_per_row = [r.columns(4) for r in rows]
        cols = [column for row in cols_per_row for column in row]

        for image_index, mon_image in enumerate(final_images):
            cols[image_index].image(mon_image)

    if st.button("About"):
     st.text("This is a Malaria detector that detects whether a given image belongs to a parasitized or uninfected cell.")
     st.text("This classifier uses VGG16, a pre-trained Convolutional Neural Network architecture.")
     st.text("This classifier has been deployed using Streamlit.")
if __name__=='__main__':
    main()
