
import streamlit as st
import numpy as np
from PIL import Image 
from tensorflow.keras.models import load_model
from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image
import base64
from pathlib import Path


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def img_to_html(img_path):
    img_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
      img_to_bytes(img_path)
    )
    return img_html


st.markdown("<p style='text-align: right; color: white;'> "+img_to_html('kpmg.png')+"</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white;'> "+img_to_html('national_emblem_resized.png')+"</p>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: blue;'>JHARKHAND HEALTH AI</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: grey;'>KPMG DEMO</h3>", unsafe_allow_html=True)
st.write("\n\n\n\n\n")
st.write("\n\n\n\n\n")
st.write("\n\n\n\n\n")


st.set_option('deprecation.showfileUploaderEncoding', False)


#@st.cache(allow_output_mutation=True)
@st.cache_resource()
def loading_model():
  fp = "cnn_pneu_vamp_model.h5"
  model_loader = load_model(fp)
  return model_loader

cnn = loading_model()
st.write("""
### X-Ray Classification [Pneumonia/Normal])""")

temp = st.file_uploader("Upload X-Ray Image")


buffer = temp
temp_file = NamedTemporaryFile(delete=False)
if buffer:
    temp_file.write(buffer.getvalue())
    st.write(image.load_img(temp_file.name))


if buffer is None:
  st.text("Oops! that doesn't look like an image. Try again.")

else:
  test_img = image.load_img(temp_file.name, target_size=(500, 500),color_mode='grayscale')

  # Preprocessing the image
  pp_img = image.img_to_array(test_img)
  pp_img = pp_img/255
  pp_img = np.expand_dims(pp_img, axis=0)

  #predict
  preds= cnn.predict(pp_img)
  if preds>= 0.5:
    out = (' {:.2%} chances that patient is suffering from Pneumonia '.format(preds[0][0]))
  
  else:
    out = ('{:.2%} chances that patient is not suffering from Pneumonia, Normal case'.format(1-preds[0][0]))

  st.success(out)
  
  image = Image.open(temp)
  st.image(image,use_column_width=True)
          
            

  

  
