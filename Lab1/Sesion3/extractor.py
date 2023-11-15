from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np

class Extractor():
    def __init__(self, weights=None):
        self.weights = weights  # so we can check elsewhere which model
        
        # create the base pre-trained model InceptionV3 with pre-trained imagenet weights including the top layers
        base_model = InceptionV3(include_top=True, weights='imagenet')
        output = base_model.get_layer('avg_pool').output

        # We'll extract features at the final pool layer.
        # Define Model inputs and outputs (Model class)
        # define base_mode.input as input and the final pool layer (features extraction) as output
        self.model = Model(inputs=base_model.input, outputs=output)

    def extract(self, image_path):        
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        ## Preprocess input 
        x = preprocess_input(x)

        #Extract the feature/prediction from the model (Model class)
        features = self.model(x)
        features = features[0]

        return features
