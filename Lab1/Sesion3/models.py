"""
A collection of models.
"""
from tensorflow.keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import TimeDistributed
from keras.layers import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
from collections import deque
import sys

class ResearchModels():
    def __init__(self, nb_classes, model, seq_length,
                 saved_model=None, features_length=2048):
        """
        `model` = lstm            
        `nb_classes` = the number of classes to predict
        `seq_length` = the length of our video sequences
        `saved_model` = the path to a saved Keras model to load
        """

        # Set defaults.
        self.seq_length = seq_length
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        self.feature_queue = deque()

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        # Get the appropriate model.
        
        print("Loading LSTM model.")
        self.input_shape = (seq_length, features_length)
        self.model = self.lstm()
        

        # Now compile the network.
        optimizer = Adam(lr=1e-5, weight_decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=metrics)

        print(self.model.summary())

    def lstm(self):
        """Build a simple LSTM network."""
        #Define an empty Sequential Model.
        model = Sequential()
        #Add a simple LSTM layer with 2048 units, do note return the last output, 
        #with self.input_shape input_shape and 0.5 dropout 
        model.add(LSTM(units=2048, input_shape=(self.input_shape), dropout=0.5))
        #Add a fully-connected layer (densely-connected NN layer, "Dense layer") with 512 units and relu activation
        model.add(Dense(units=512, activation='relu'))
        #Add a Dropout layer with 0.5
        model.add(Dropout(0.5))
        #Add a logistic layer (densely-connected NN layer, "Dense layer") with the number of video classes units (self.nb_classes) and softmax activation
        model.add(Dense(units=self.nb_classes, activation='softmax'))

        return model

    
