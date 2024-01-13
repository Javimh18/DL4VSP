import numpy as np
import torch
import torch.nn.functional as F

import motmetrics as mm
from torchvision.ops.boxes import clip_boxes_to_image, nms
from munkres import Munkres, print_matrix
from kalman_filter import KalmanFilter_XYHW

DISASSOCIATE = 1e9
WARM_UP = 10

def _process_nans(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if np.isnan(matrix[i][j]):
                matrix[i][j] = DISASSOCIATE
                
    return matrix

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
        
    def reset(self, hard=True):
        self.tracks = []
        
        if hard:
            self.track_num = 0
            self.results = {}
            self.im_index = 0

    def add(self, new_boxes, new_scores):
        """
        Initializes new Track objects and saves them.
        """
        num_new = len(new_boxes)
        for i in range(num_new):
            self.tracks.append(
                Track(
                    new_boxes[i],
                    new_scores[i],
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
            box = torch.stack([t.box for t in self.tracks], 0)
        else:
            box = torch.zeros(0).cuda()
            
        return box
    
    def data_association(self, boxes, scores):
        if self.tracks:
            track_ids = [t.id for t in self.tracks]
            track_boxes = np.stack([t.box.numpy() for t in self.tracks], axis=0)
            track_kf_estimations = np.stack([t.kf_estimations.numpy() for t in self.tracks], axis=0)
            track_kfs = [t.kf for t in self.tracks]
            
            # compute the distance based on IoU (distance=1-IoU)
            distance = mm.distances.iou_matrix(track_boxes, boxes.numpy(), max_iou=0.5)
            
            for i in range(len(distance)):
                for j in range(len(distance[0])):
                    if np.isnan(distance[i][j]):
                        distance[i][j] = DISASSOCIATE
            
            # if there are more tracks than detected objects, we traspose the matrix and
            # process it in the "inversed" way:
            #   distance.shape -> tracks, detections
            #   distance.T.shape -> detections, tracks
            t_flag = False
            shape = distance.shape
            if shape[0] > shape[1]:
                distance = np.transpose(distance)
                t_flag = True
                
            # Hungarian Algorithm computation
            indexes = self.hung_algo.compute(distance)
            for as_idx in indexes:
                if t_flag:
                    d,t = as_idx[0], as_idx[1] # detections, tracks
                else:
                    t,d = as_idx[0], as_idx[1] # tracks, detections
                    
                self.tracks[t].box = boxes[d]
                
                
            # removing lost tracks
            remove_tracks_id = []
            if t_flag:
                distance_t = np.transpose(distance)
            else:
                distance_t = distance
                
            for t,dist in zip(self.tracks, distance_t):
                # Numpy transpose is quite buggy... 
                if np.all(dist>DISASSOCIATE-100): # no detection associated to a track
                    print("removed track!")
                    remove_tracks_id.append(t.id)
            self.tracks = [t for t in self.tracks\
                    if t.id not in remove_tracks_id]
            
            # adding new tracks iterating by the detections
            new_boxes = []
            new_scores = []
            if t_flag:
                distance_t = distance
            else:
                distance_t = np.transpose(distance)
            for d,dist in enumerate(distance_t):
                if np.all(dist>DISASSOCIATE-100): # no track associated w/ detection, so add new track as the detection
                    print("Added new track!")
                    new_boxes.append(boxes[d])
                    new_scores.append(scores[d])
            self.add(new_boxes, new_scores)
            
        else:
            self.add(boxes, scores)
            
            
            
    def step(self, frame):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        # object detection
        boxes, scores = self.obj_detect.detect(frame['img'])
        self.data_association(boxes, scores)

        # results
        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            self.results[t.id][self.im_index] = np.concatenate([t.box.cpu().numpy(), np.array([t.score])])

        self.im_index += 1

    def get_results(self):
        return self.results
    
    @staticmethod
    def _compute_miou_distances(self, boxes, tracks, kalman_estimations):
        distance_tracks = mm.distances.iou_matrix(tracks, boxes.numpy(), max_iou=0.5)
        distance_kalman = mm.distances.iou_matrix(kalman_estimations, boxes.numpy(), max_iou=0.5)
        
        distance_tracks = _process_nans(distance_tracks)
        distance_kalman = _process_nans(distance_kalman)
        
        # first we let the Kalman filter warm-up, so we do not take into account its predictions
        # but as frames are processed, we give more importance to its predictions.
        l = 0
        if self.im_index > 3*WARM_UP:
            l = 1
        elif self.im_index > 2*WARM_UP:
            l = 0.9
        elif self.im_index > WARM_UP:
            l = 0.7
        
        # pondered result depending on l value
        miou_pondered = (1-l)*distance_tracks + l*distance_kalman
        return miou_pondered
    
               
class Track(object):
    """
    This class contains all necessary fields/data for every individual track.
    """
    def __init__(self, box, score, track_id):
        self.box = box
        self.score = score
        self.id = track_id
        self.kf = KalmanFilter_XYHW()
        self.kl_estimations = self.kf.detect(box[0], box[1], box[2], box[3])
