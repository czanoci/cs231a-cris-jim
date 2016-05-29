import sys
import numpy as np
from constants import *

class DSFNode:
	def __init__(self, i):
		self.clusterSize = 1
		self.parent = self
		self.pieceIndex = i
		self.localRot = 0
		self.localCoords = np.array([[0],[0]])

	# def get_parent(self):
	# 	return self.parent

	# def get_clusterSize(self):
	# 	return self.clusterSize

	# def set_parent(self, p):
	# 	self.parent = p

	# def increment_clusterSize(self, cs):
	# 	self.clusterSize += cs

	# def get_pieceIndex(self):
	# 	return self.data


class DisjointSetForest:
	def __init__(self, numNodes):
		self.numClusters = numNodes
		self.nodes = []
		self.pieceCoordMap = {}
		for i in xrange(numNodes):
			self.nodes.append(DSFNode(i))
			self.pieceCoordMap[i] = np.array([[0],[0]])

	def rotMat(self, r):
		if r == 0:
			return np.identity(2)
		elif r == 1:
			return np.array([[0, -1], [1, 0]])
		else:
			return -self.rotMat(r-2)

	# Given the index of a node, finds the cluster representative for that node.
	# Compresses paths as it goes
	def find(self, i):
		return self.find_node(self.nodes[i])

	# returns the representative, as well as the local rotation and local coordinates of the input node
	# note that because we are doing path compression, these are with respect to the representative
	def find_node(self, n):
		if n.parent != n:
			rep, parRot, parCoords = self.find_node(n.parent)
			if n.parent != rep:
				n.parent = rep
				n.localRot = (n.localRot + parRot) % 4
				n.localCoords = parCoords + np.dot(self.rotMat(parRot), n.localCoords)

		return n.parent, n.localRot, n.localCoords

	# Merges the clusters holding the nodes at index i and j.
	# Returns True if it made a change, False if they were already in same cluster
	def union(self, i, j, edgeNum_i, edgeNum_j):
		rep_i, rot_i, coords_i = self.find(i)
		rep_j, rot_j, coords_j = self.find(j)

		# check that the pieces aren't already in the same cluster
		if rep_i == rep_j:
			return -1

		# originally, the numbers passed for the edgeNum aren't correct for j, since it is really the number
		# of rotations needed to apply to the piece, not the edge number.  This corrects it and makes sure that
		# the encoding of the edge numbers is consistent
		# edgeNum_j = (edgeNum_j + 2) % 4

		# determine the small and the big cluster
		if rep_i.clusterSize >= rep_j.clusterSize:
			clust_big = rep_i
			clust_small = rep_j
			piece_big = self.nodes[i]
			piece_small = self.nodes[j]
			piece_rot_big = edgeNum_i
			piece_rot_small = edgeNum_j
			big_is_left = True
		else:
			clust_big = rep_j
			clust_small = rep_i
			piece_big = self.nodes[j]
			piece_small = self.nodes[i]
			piece_rot_big = edgeNum_j
			piece_rot_small = edgeNum_i
			big_is_left = False

		# compute the amount we will have to rotate the small cluster to orient it the same as the large
		# small_clust_rot = (piece_rot_small - piece_rot_big + piece_big.localRot - piece_small.localRot) % 4
		small_clust_rot = (piece_rot_small - piece_rot_big + piece_big.localRot - piece_small.localRot) % 4
		# is this better?? piece_small_switch = (-piece_rot_big + piece_big.localRot) % 4 if big_is_left else (-piece_rot_big + 2 + piece_big.localRot) % 4
		piece_small_switch = (-piece_rot_big + piece_big.localRot) % 4 if big_is_left else (-piece_rot_big + 2 + piece_big.localRot) % 4

		if piece_small_switch == 0:
			offset = np.array([[1], [0]])
		elif piece_small_switch == 1:
			offset = np.array([[0], [1]])
		elif piece_small_switch == 2:
			offset = np.array([[-1], [0]])
		else:
			offset = np.array([[0], [-1]])


		small_rot_mat = self.rotMat(small_clust_rot)
		small_clust_trans = piece_big.localCoords + offset - np.dot(small_rot_mat, piece_small.localCoords)


		coords_small = self.pieceCoordMap[clust_small.pieceIndex]
		new_coords_small = np.dot(small_rot_mat, coords_small) + small_clust_trans
		coords_big = self.pieceCoordMap[clust_big.pieceIndex]
		is_disjoint = set(map(tuple, new_coords_small.T)).isdisjoint(map(tuple, coords_big.T))

		if not is_disjoint:
			return -1

		clust_small.localCoords = small_clust_trans
		clust_small.localRot = small_clust_rot
		clust_small.parent = clust_big
		clust_big.clusterSize += clust_small.clusterSize
		self.pieceCoordMap[clust_big.pieceIndex] = np.concatenate((coords_big, new_coords_small), axis = 1)
		del self.pieceCoordMap[clust_small.pieceIndex]
		self.numClusters -= 1
		
		return clust_big.pieceIndex

	def reconstruct(self, clustIndex, pieces):
		# Need to get pieces
		rep, _, _ = self.find(clustIndex)

		coords = self.pieceCoordMap[rep.pieceIndex]
		min_x = min(coords[0, :])
		min_y = min(coords[1, :])
		max_x = max(coords[0, :])
		max_y = max(coords[1, :])

		H = (max_y - min_y + 1) * P
		W = (max_x - min_x + 1) * P
		img = np.zeros([H, W, 3])

		#offset = np.array([[-min_x],[-min_y]])
		for node in self.nodes:
			i = node.pieceIndex
			i_rep, _, _ = self.find(i)
			if i_rep != rep:
				continue

			sq = pieces[i]
			pixels = sq.get_rotated_pixels(node.localRot)
			x = node.localCoords[0, 0] - min_x
			y = node.localCoords[1, 0] - min_y

			img[(H - (y + 1)*P):(H - y*P), x*P:(x + 1)*P, :] = pixels
		return img





