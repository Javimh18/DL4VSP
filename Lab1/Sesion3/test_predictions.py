from data import DataSet
import time
import os.path
from Sesion2.train_cnn import get_generators, get_model
from models import ResearchModels

class_limit = 5  # int, can be 1-101 or None
seq_length = 5
image_shape = None
data_type_cnn = "images"
data_type_lstm = "features"
model_lstm = 'lstm'
saved_model_lstm = './data/checkpoints/lstm-5-features.008-0.211.hdf5'# TODO: Get path of the best model in the LSTM
saved_model_cnn = './data/checkpoints/inception.001-0.37.hdf5'# TODO: Get path of the best model in the cnn
batch_size = 8

if __name__ == '__main__':
    
    # Get the data for the lstm and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )
        
    # get generator from the lstm features
    val_generators_lstm = data.frame_generator(batch_size, 'test', data_type_lstm)
    
    # get the data from the cnn
    _, val_generators_cnn = get_generators()
    
    # get InceptionV3 model and load the best weights stored in checkpoints
    cnn_model = get_model(weights=saved_model_cnn)
    
    # get RNN model and load the best weights stored in checkpoints
    rm = ResearchModels(len(data.classes), model_lstm, seq_length, saved_model_lstm)
    
    # loop through the val generator for the cnn and get the misclassifications
    misclassfications_cnn = []
    for idx, batch in enumerate(val_generators_cnn):
        for image, label in batch:
            pred_label = cnn_model(image)
            if pred_label != label:
                misclassfications_cnn.append(image, label, pred_label)
                
    # loop through the X test and y test data for the lstm and get the misclassifications
    
    features_paths = []
    for row in data.data:
        if row[0] == 'test':
            filename = row[2]
            path = os.path.join(data.sequence_path, filename + '-' + str(data.seq_length) + \
            '-' + data_type_lstm + '.npy')
            features_paths.append(features_paths)
            
    misclassfications_lstm = []
    for idx, batch in enumerate(val_generators_lstm):
        for features, label in batch:
            pred_label = rm.model(features)
            if pred_label != label:
                misclassfications_lstm.append(features, label, pred_label)
        
