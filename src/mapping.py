import networkx as nx #version: 2.4
import graphviz
import pydot
import re

import matplotlib.pyplot as plt

%matplotlib inline

# a traditional graph
graphdataset = nx.Graph(nx.drawing.nx_pydot.read_dot("gd.gv"))

#change the weights into ints
for i in graphdataset.edges:
    graphdataset.edges[i]['weight']=int(re.findall(r'[0-9]+', (graphdataset.edges[i]['weight']))[0])

# draw the graph 
# nx.spring_layout calculate the position by forced directed algorithm
nx.draw(graphdataset,nx.spring_layout(graphdataset))
# you may need to change the canvas size to see the edges
