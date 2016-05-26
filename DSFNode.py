

class DSFNode:
	def __init__(self, d):
		self.order = 0
		self.parent = self
		self.data = d

	def get_parent(self):
		return self.parent

	def get_order(self):
		return self.order

	def set_parent(self, p):
		self.parent = p

	def increment_order(self):
		self.order += 1

	def get_data(self):
		return self.data

class DSFEdge:
	def __init__(self, i1, i2, r1, r2):
		self.ind1 = i1
		self.ind2 = i2
		self.rot1 = r1
		self.rot2 = r2

	def get_data(self):
		return (self.ind1, self.ind2, self.rot1, self.rot2)


class DisjointSetForest:
	def __init__(self, numNodes):
		self.numClusters = numNodes
		self.nodes = []
		for i in xrange(numNodes):
			self.nodes.append(DSFNode(i))

	# Given the index of a node, finds the cluster representative for that node.
	# Compresses paths as it goes
	def find(self, i):
		return self.find_node(self.nodes[i])

	def find_node(self, n):
		if n.get_parent() == n:
			return n

		n.set_parent(self.find_node(n.get_parent()))
		return n.get_parent()

	# Merges the clusters holding the nodes at index i and j.
	# Returns True if it made a change, False if they were already in same cluster
	def union(self, i, j):
		rep_i = self.find(i)
		rep_j = self.find(j)

		if rep_i == rep_j:
			return False

		if rep_i.get_order() > rep_j.get_order():
			rep_j.set_parent(rep_i)
		elif rep_j.get_order() > rep_i.get_order():
			rep_i.set_parent(rep_j)
		else:
			rep_j.set_parent(rep_i)
			rep_i.increment_order()

		self.numClusters -= 1
		return True

	def get_num_clusters(self):
		return self.numClusters