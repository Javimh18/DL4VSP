import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops.boxes import clip_boxes_to_image, nms
import torchvision.transforms.functional as TF
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms

import motmetrics as mm
from munkres import Munkres, print_matrix
import cv2
import numpy as np
import time
# from kalman_filter import KalmanFilter as KF

DISASSOCIATE = 1e9
WARM_UP = 4
PATIENCE_RMV = 10
PATIENCE_INIT = 0
DELTA_T_KALMAN = 1 # increments in time between frames (Pondered mean of all the frame rates of the dataset)
ACC = 1

FEAT_LEN = 2048
DIM_FEAT_INPUT = 224
MEAN_IM = [0.485, 0.456, 0.406]
STD_IM = [0.229, 0.224, 0.225]
MAX_DESCRIPTORS = 3

def _process_nans(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if np.isnan(matrix[i][j]):
                matrix[i][j] = DISASSOCIATE
                
    return matrix

def _crop_box(image, box):
    x0, y0, x1, y1 = box
    return image.crop((x0, y0, x1, y1))

def _compose(frame, bboxes):
    frame_pil = TF.to_pil_image(frame)
    # x0, y0, x1, y1 -> x, y, w, h
    crops = list(map(lambda box: _crop_box(frame_pil, box), bboxes))
    crop_resized = list(map(lambda crop: TF.resize(crop, [DIM_FEAT_INPUT, DIM_FEAT_INPUT]), crops))
    crop_tensors = list(map(lambda crop: transforms.ToTensor()(crop), crop_resized))
    crops = torch.stack(crop_tensors, dim=0)
    norm_tensor = TF.normalize(crops, mean=MEAN_IM, std=STD_IM)
    
    return norm_tensor

class Tracker:
    """
    The main tracking file, here is where the magic happens.
    """
    
    def __init__(self, obj_detect):
        self.obj_detect = obj_detect
        
        self.tracks = []
        self.track_num = 0
        self.im_index = 0
        self.results = {}
        self.hung_algo = Munkres()
        self.mot_accum = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # FEATURE EXTRACTOR FOR DEEP SIMILARITY DISTANCE
        # Load pre-trained ResNet-50 model
        resnet50 = models.resnet50(pretrained=True)
        # Remove the classifier (fully connected) layer
        self.feat_extractor = nn.Sequential(*list(resnet50.children())[:-1])
        self.feat_extractor.to(self.device)
        # set grad to none for better performance
        for param in self.feat_extractor.parameters():
            param.grad = None
        # put the model in eval mode
        self.feat_extractor.eval()
        
        
    def reset(self, hard=True):
        self.tracks = []
        
        if hard:
            self.track_num = 0
            self.results = {}
            self.im_index = 0

    def add(self, new_boxes, new_scores, feat_descs):
        """
        Initializes new Track objects and saves them.
        """
        num_new = len(new_boxes)
        for i in range(num_new):
            self.tracks.append(
                Track(
                    new_boxes[i],
                    new_scores[i],
                    feat_descs[i],
                    self.track_num + i
                )
            )
        self.track_num += len(new_boxes)

    def get_pos(self):
        """
        Get positions of all active tracks
        """
        
        if len(self.tracks) == 1:
            box = self.tracks[0].box
        elif len(self.tracks) > 1:
            box = torch.stack([t.box for t in self.tracks if t.is_active == True], 0)
        else:
            box = torch.zeros(0).cuda()
            
        return box
    
    def data_association(self, boxes, scores, frame):
        max_iou = 0.5
        if self.tracks:
            track_boxes = np.stack([t.box.numpy() for t in self.tracks], axis=0)
            track_kf_estimation = np.stack([t.kf_estimation for t in self.tracks], axis=0)

            # compute the distance based on IoU (distance=1-IoU)
            start_t = time.time()
            distance = self._compute_distances(boxes.numpy(), track_boxes, track_kf_estimation, frame, max_iou)
            
            # if there are more tracks than detected objects, we traspose the matrix and
            # process it in the "inversed" way:
            #   distance.shape -> tracks, detections
            #   distance.T.shape -> detections, tracks
            t_flag = False
            shape = distance.shape
            if shape[0] > shape[1]:
                distance = distance.T
                t_flag = True
                
            # Hungarian Algorithm computation
            indexes = self.hung_algo.compute(distance)
            for as_idx in indexes:
                if t_flag:
                    d,t = as_idx[0], as_idx[1] # detections, tracks
                else:
                    t,d = as_idx[0], as_idx[1] # tracks, detections
                    
                self.tracks[t].box = boxes[d]
                self.tracks[t].n_frames_track += 1
                self.tracks[t].frames_consecutive_track += 1
                self.tracks[t].frames_since_recent_track = 0
                
                if self.im_index <= PATIENCE_INIT: # At the beginning we consider all detections tracks
                    self.tracks[t].is_active = True
                    self.tracks[t].kf_estimation = self.tracks[t].kf.detect(boxes[d][0],
                                                                        boxes[d][1],
                                                                        boxes[d][2],
                                                                        boxes[d][3])
        
                elif self.im_index > PATIENCE_INIT and self.tracks[t].frames_consecutive_track > PATIENCE_INIT:
                    self.tracks[t].is_active = True
                    # only update the kalman filter estimations if and only if the 
                    # the track is active and it has passed the PATIENCE INIT value
                    self.tracks[t].kf_estimation = self.tracks[t].kf.detect(boxes[d][0],
                                                                        boxes[d][1],
                                                                        boxes[d][2],
                                                                        boxes[d][3])
                    
            # for better inference times
            with torch.no_grad():
                # get the features from the feature extractor and append them to their correspondent tracks
                updated_boxes_in_frame = _compose(frame.squeeze(0), boxes.numpy())
                updated_boxes_in_frame = updated_boxes_in_frame.to(self.device)
                feat_updated_boxes = self.feat_extractor(updated_boxes_in_frame)
                for as_idx in indexes:
                    if t_flag:
                        d,t = as_idx[0], as_idx[1] # detections, tracks
                    else:
                        t,d = as_idx[0], as_idx[1] # tracks, detections
                        
                    if len(self.tracks[t].descriptors) == MAX_DESCRIPTORS:
                        self.tracks[t].descriptors.pop(0)
                    self.tracks[t].descriptors.append(feat_updated_boxes[d])
                    
            # create the cost matrix with all np.inf except assign "0" for the 
            # indexes that the hungarian algo associated
            distance = np.full_like(distance, DISASSOCIATE)
            for as_idx in indexes:
                distance[as_idx] = 0
            
            # compiling lost tracks
            remove_tracks_id = []
            if t_flag:
                distance_t = distance.T
            else:
                distance_t = distance
            for t,dist in zip(self.tracks, distance_t):
                # Numpy transpose is quite buggy... 
                if np.all(dist>DISASSOCIATE-10): # no detection associated to a track
                    if t.frames_since_recent_track == PATIENCE_RMV: # if patience reached remove track
                        remove_tracks_id.append(t.id)
                    else:
                        t.frames_since_recent_track += 1
                        t.frames_consecutive_track == 0
                        # we update the kf_estimation just in case the tracker
                        # recovers sight of the lost track in the following detections
                        t.kf_estimation = t.kf.detect(t.kf_estimation[0],
                                                       t.kf_estimation[1],
                                                       t.kf_estimation[2],
                                                       t.kf_estimation[3])
                        
            # removing tracks in the remove_tracks_id array
            self.tracks = [t for t in self.tracks\
                    if t.id not in remove_tracks_id]
            
            # adding new tracks iterating by the detections
            new_boxes = []
            new_scores = []
            new_feats = []
            if t_flag:
                distance_t = distance
            else:
                distance_t = distance.T
            for d,dist in enumerate(distance_t):
                if np.all(dist>DISASSOCIATE-10): 
                    # no track associated w/ detection, so add new track as the detection
                    new_boxes.append(boxes[d])
                    new_scores.append(scores[d])
            
            if len(new_boxes) > 0:
                with torch.no_grad():
                    tensor_new_boxes = torch.stack(new_boxes, dim=0)
                    # get the features from the feature extractor
                    new_boxes_in_frame = _compose(frame.squeeze(0), tensor_new_boxes.numpy())
                    new_boxes_in_frame = new_boxes_in_frame.to(self.device)
                    tensor_new_feats = self.feat_extractor(new_boxes_in_frame)
                    for i in range(tensor_new_feats.size(0)):
                        new_feats.append(tensor_new_feats[i])
            
            self.add(new_boxes, new_scores, new_feats)
            
        else:
            # for better inference times
            with torch.no_grad():
                start_t_add = time.time()
                cropped_boxes = []
                cropped_boxes = _compose(frame.squeeze(0), boxes.numpy())
                cropped_boxes = cropped_boxes.to(self.device)
                feat_cropped_boxes = self.feat_extractor(cropped_boxes)
                features = []
                for i in range(feat_cropped_boxes.size(0)):
                    features.append(feat_cropped_boxes[i])
                self.add(boxes, scores, features)
            
            
    def step(self, frame):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        # object detection
        boxes, scores = self.obj_detect.detect(frame['img'])
        self.data_association(boxes, scores, frame['img'])

        # results
        for t in self.tracks:
            if t.id not in self.results.keys(): 
                self.results[t.id] = {}
            
            # only return those tracks that are active
            if t.is_active == True:
                self.results[t.id][self.im_index] = np.concatenate([t.box.cpu().numpy(), np.array([t.score])])

        self.im_index += 1
        # set all of them to False again
        for t in self.tracks:
            t.is_active = False


    def get_results(self):
        return self.results
    
    
    def _compute_distances(self, boxes, tracks, kalman_estimations, frame, max_iou=0.5):
        distance_tracks = mm.distances.iou_matrix(tracks, boxes, max_iou)
        distance_kalman = mm.distances.iou_matrix(kalman_estimations, boxes, max_iou)
        
        distance_tracks = _process_nans(distance_tracks)
        distance_kalman = _process_nans(distance_kalman)
        
        # first we let the Kalman filter warm-up, so we do not take into account its predictions
        # but as frames are processed, we give more importance to them.
        l_1 = np.zeros(distance_kalman.shape[0])
        l_2 = 0.3
        
        for i,t in enumerate(self.tracks):
            if t.n_frames_track >= int(1.5*WARM_UP):
                l_1[i] = 0.9
            elif t.n_frames_track >= WARM_UP:
                l_1[i] = 0.5
        
        # pondered result depending on l value
        distance_track = (1-l_1[:, np.newaxis])*distance_tracks + l_1[:, np.newaxis]*distance_kalman
        distance_app = self._compute_appearance_sim(boxes, frame)
        return l_2*distance_track + (1-l_2)*distance_app
    
               
    def _compute_appearance_sim(self, boxes, frame):
        # for better inference times
        with torch.no_grad():
            crop_boxes = _compose(frame.squeeze(0), boxes)
            c_box = crop_boxes.to(self.device)
            box_feat = self.feat_extractor(c_box)
        
        # create a matrix with the shape (n_tracks, detections)
        d = np.full((len(self.tracks), len(boxes)), np.inf)
        for i, t in enumerate(self.tracks):
            descriptors = torch.stack(t.descriptors, dim=0)
            for j in range(box_feat.size(0)):
                d_min = np.inf
                d_cos = 1 - F.cosine_similarity(box_feat[j].unsqueeze(0), descriptors, dim=1)
                d_cos = torch.min(d_cos).item()
                if d_cos < d_min:
                    d_min = d_cos
                # once we've got the minimum distance between all descriptors
                d[i,j] = d_min
        
        return d   
          
            
class Track(object):
    """
    This class contains all necessary fields/data for every individual track.
    """
    def __init__(self, box, score, feat_desc, track_id):
        self.box = box
        self.score = score
        self.id = track_id
        self.kf = KalmanFilter_XYHW()
        self.kf_estimation = self.kf.detect(box[0], box[1], box[2], box[3])
        self.is_active = False
        self.frames_since_recent_track = 0 # How many frames have passed since the track was detected
        self.frames_consecutive_track = 0 # How many frames have been the track consecutively detected
        self.n_frames_track = 0 # In how many frames from the sequence has the track been in
        self.descriptors = [feat_desc] # Array with the descriptor tensor of the track (we keep the last 100 descriptors to compare)
        
        
class KalmanFilter_XYHW(object):
    def __init__(self):
        self.kf_xy = KalmanFilter()
        self.kf_hw = KalmanFilter()
        
    def detect(self, c_X, c_Y, c_H, c_W):
        x, y = self.kf_xy.predict(c_X, c_Y)
        h, w = self.kf_xy.predict(c_H, c_W)
        return np.array([x,y,h,w])[:, 0]
    
    
class KalmanFilter(object):
    def __init__(self):
        self.kf = cv2.KalmanFilter(4,2)
        # A matrix in the priori estimations (state transition)
        self.kf.transitionMatrix = np.array([[1, 0, DELTA_T_KALMAN**2, 0],
                                             [0, 1, 0, DELTA_T_KALMAN**2],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], dtype=np.float32)
        
        # H matrix in the measure estimations
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], dtype=np.float32)
        
        # Q matrix, the initial process noise covariance
        self.kf.processNoiseCov = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [1, 0, 1, 0],
                                            [0, 1, 0, 1]], dtype=np.float32) * ACC**2
        
        # R matrix, the initial measurements noise covariance
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32)
        
        
    def predict(self, coord1, coord2):
        '''
            Point estimation of the next point using Kalman predict and correct
        '''
        
        measured = np.array([[np.float32(coord1)], [np.float32(coord2)]])
        self.kf.correct(measured) # first, we correct the KF with the new measurement of the bbox coordinate
        predicted = self.kf.predict() # then, we predict the bbox coordinate for the next frame
        c_1, c_2 = predicted[0], predicted[1] 
        return c_1, c_2