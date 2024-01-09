import numpy as np
import torch
import torch.nn.functional as F

import motmetrics as mm
from torchvision.ops.boxes import clip_boxes_to_image, nms
from munkres import Munkres, print_matrix


class Tracker:
	"""The main tracking file, here is where magic happens."""

	def __init__(self, obj_detect):
		self.obj_detect = obj_detect

		self.tracks = []
		self.track_num = 0
		self.im_index = 0
		self.results = {}

		self.mot_accum = None

	def reset(self, hard=True):
		self.tracks = []

		if hard:
			self.track_num = 0
			self.results = {}
			self.im_index = 0

	def add(self, new_boxes, new_scores):
		"""Initializes new Track objects and saves them."""
		num_new = len(new_boxes)
		for i in range(num_new):
			self.tracks.append(Track(
				new_boxes[i],
				new_scores[i],
				self.track_num + i
			))
		self.track_num += num_new

	def get_pos(self):
		"""Get the positions of all active tracks."""
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

			# compute distance based on IoU (distance=1-IoU)
			distance = mm.distances.iou_matrix(track_boxes, boxes.numpy(), max_iou=0.5)
		
			for i in range(len(distance)):
				for j in range(len(distance[0])):
					if np.isnan(distance[i][j]) == True:
						distance[i][j] = np.inf

			print(distance.shape)
			# Hungarian Algo
			m = Munkres()
			indexes = m.compute(distance)

			# update existing tracks
			remove_track_ids = []
			for t, dist in zip(self.tracks, distance):
				if np.isinf(dist).all():
					remove_track_ids.append(t.id)
				else:
					match_id = np.nanargmin(dist)
					t.box = boxes[match_id]
			self.tracks = [t for t in self.tracks
					if t.id not in remove_track_ids]

			# add new tracks
			new_boxes = []
			new_scores = []
			for i, dist in enumerate(np.transpose(distance)):
				if np.isinf(dist).all():
					new_boxes.append(boxes[i])
					new_scores.append(scores[i])
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


class Track(object):
	"""This class contains all necessary for every individual track."""

	def __init__(self, box, score, track_id):
		self.id = track_id
		self.box = box
		self.score = score
