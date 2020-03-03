import community
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

# a traditional graph
graphdataset = nx.Graph(nx.drawing.nx_pydot.read_dot("gd.gv"))

#change the weights into ints
for i in graphdataset.edges:
    graphdataset.edges[i]['weight']=int(re.findall(r'[0-9]+', (graphdataset.edges[i]['weight']))[0])

# set coordinates for vertices using force directed algorithm
#size = float(len(set(partition.values())))
pos = nx.spring_layout(graphdataset, k=1/math.pow(len(graphdataset), 0.3))

# stack all coordinates for Voronoi
coor_lis = []
#point_lis = []
for p in range(0, len(pos.keys())):
    #point_lis.append(str(p))
    coor_lis.append(pos[str(p)])

#print(len(coor_lis))
points = np.stack(coor_lis)


# create Voronoi object
vor = Voronoi(points, furthest_site=False)
#ridge_lis = vor.ridge_points.tolist()
"""
# compute the best partition into communities
partition = community.best_partition(graphdataset)

commu = {}
for p, com in partition.items():
    if(com not in commu): commu[com] = [p]
    else: commu[com].append(p)

ridge_lis = vor.ridge_points.tolist()
#print(len(ridge_lis))

for p_lis in commu.values():
    for i in range(0, len(p_lis)-1):
        try: ridge_lis.remove([int(p_lis[i]), int(p_lis[i+1])])
        except: 
            try: ridge_lis.remove([int(p_lis[i+1]), int(p_lis[i])])
            except: continue

#print(len(ridge_lis))
vor.ridge_points = np.array(ridge_lis)
"""
"""
ridge_lis = vor.ridge_points.tolist()
del ridge_lis[0]
del ridge_lis[1]
vor.ridge_points = np.array(ridge_lis)
"""

# plot Voronoi diagram
voronoi_plot_2d(vor, show_vertices=False)

plt.show()
