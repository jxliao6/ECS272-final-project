import graphviz
import pydot
import re
import math
import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import community

def solver(graphdataset, pos):
	fig = plt.figure(figsize=(9, 9))
	#graphdataset = nx.Graph(nx.drawing.nx_pydot.read_dot("univ.gv"))
	partition = community.best_partition(graphdataset)
	#pos = nx.spring_layout(graphdataset, k=1/math.pow(len(graphdataset), 0.3),scale=10)

	nx.draw(graphdataset,pos=pos,scale=10,\
        with_labels=True,node_shape="s",node_size=450,edge_color="grey",alpha=1, \
			node_color=list(partition.values()), cmap=plt.cm.Set3)

	return fig

if __name__ == '__main__':
	fig = solver(graphdataset, pos, partition)
	plt.show()
	