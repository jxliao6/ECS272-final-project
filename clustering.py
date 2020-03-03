import community
import graphviz
import pydot
import re
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random
import math

random.seed()
# a traditional graph
graphdataset = nx.Graph(nx.drawing.nx_pydot.read_dot("gd.gv"))

# change the weights into ints
for i in graphdataset.edges:
    graphdataset.edges[i]['weight']=int(re.findall(r'[0-9]+', (graphdataset.edges[i]['weight']))[0])

G = graphdataset

# compute the best partition
partition = community.best_partition(G, weight='weight')
color_list = ['steelblue', 'black', 'yellow', 'seagreen', 'cyan', 'darkblue', 'darkseagreen', \
    'lightcoral', 'red', 'lightblue', 'hotpink', 'gray', 'darkviolet', 'deeppink', 'darkred', \
        'darksalmon', 'darkgreen', 'darkgoldenrod', 'orange', 'purple']

# drawing
size = float(len(set(partition.values())))
pos = nx.spring_layout(G, k=1/math.pow(len(graphdataset), 0.3))
count = 0.
for com in set(partition.values()):
    ci = random.randint(0, len(color_list)-1)
    count = count + 1.
    list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
    nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
                                node_color = color_list[ci])


nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()