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
from community import community_louvain
from matplotlib.patches import Polygon




def construct_graph(fname, embedding='force-directed'):
    """
    read the graph from .gv file
    and construct the graph with the embedding algorithm
    ------------------------------------------------------
    input
    fname: str, file name
    embedding: str, embedding algorithm
    ------------------------------------------------------
    returns
    graphdataset: nx.Graph object
    pos: dict, {node(str): coordinates(np.array)}
    """
    # a traditional graph
    graphdataset = nx.Graph(nx.drawing.nx_pydot.read_dot(fname))

    #change the weights into ints
    for i in graphdataset.edges:
        graphdataset.edges[i]['weight']=int(re.findall(r'[0-9]+', (graphdataset.edges[i]['weight']))[0])

    # set coordinates for vertices using the corresponding embedding algorithm
    if(embedding == 'force-directed'):
        pos = nx.spring_layout(graphdataset, k=1/math.pow(len(graphdataset), 0.3))

    return graphdataset, pos

def node_clustering(graph, algorithm='modularity'):
    """
    clustring all nodes with selected algorithm
    ------------------------------------------------------
    input
    graph: nx.Graph object
    algorithm: clustering algorithm
    ------------------------------------------------------
    returns
    partition: dict, {node(str): community(int)}
    """
    # compute the best partition into communities
    if(algorithm == 'modularity'):
        partition = community_louvain.best_partition(graph)

    return partition

def merge_vor(vor, com_points):
    representation = int(com_points[0])
    # remove ridge between points from the same community
    ridge_points_lis = vor.ridge_points.tolist()
    pair_remove = []
    for pair in ridge_points_lis:
        if(str(pair[0]) in com_points and str(pair[1]) in com_points):
            pair_remove.append(pair)
    for pair in pair_remove:
        ridge_points_lis.remove(pair)
    vor.ridge_points = np.array(ridge_points_lis)

    # find regions/cells need to be merged
    point_region_lis = vor.point_region.tolist()
    region_lis = []
    for p in com_points:
        region_lis.append(vor.regions[point_region_lis[int(p)]])

    # remove ridge between voronoi vertices according to the region_lis
    for i in range(0, len(region_lis)-1):
        for j in range(i+1, len(region_lis)):
            vertice_lis = []
            for v in region_lis[i]:
                if(v in region_lis[j]): vertice_lis.append(v)
            if(len(vertice_lis) == 0): continue
            assert len(vertice_lis) == 2, "More than two common vertices between two region"
            try: vor.ridge_vertices.remove(vertice_lis)
            except:
                v1, v2 = vertice_lis
                vor.ridge_vertices.remove([v2, v1])
    #print(vor.ridge_vertices)
    # merge together all regions in region_lis 
    # by adding vertices from other regions to the remaining region
    for i in range(1, len(region_lis)):
        for v in region_lis[i]:
            if (v not in region_lis[0]):
                region_lis[0].append(v)
    vor.regions[point_region_lis[int(com_points[0])]] = region_lis[0]
    for i in range(1, len(region_lis)):
        vor.regions.remove(region_lis[i])
    
    region_index_remove = []
    for i in range(1, len(com_points)):
        region_index_remove.append(point_region_lis[int(com_points[i])])
    for i in region_index_remove:
        point_region_lis.remove(i)
    vor.point_region = np.array(point_region_lis)

    colorRegions(vor)
    
    voronoi_plot_2d(vor, show_vertices=False)
    plt.show()
    return vor

def draw_vor(pos, partition):

    # stack coordinates of nodes in coor_lis to draw the Voronoi Diagram
    # indices of coor_lis correspond to the node
    coor_lis = []
    for p in range(0, len(pos.keys())):
        coor_lis.append(pos[str(p)])
    vor = Voronoi(np.stack(coor_lis))
    
    # build dictionary to store community info
    # com_node = {community: [nodes in community]}
    com_node = {}
    for p, com in partition.items():
        com_node.setdefault(com, []).append(p)

    #print(vor.regions)

    # merge voronoi cells for each community
    for p_lis in com_node.values():
        vor = merge_vor(vor, p_lis)

    #coloring

    # plot Voronoi diagram
    
    voronoi_plot_2d(vor, show_vertices=False)
    plt.show()
    
def colorRegions(vor):
    for region in vor.regions:
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            #plt.fill(*zip(*polygon), color = "green")
            plt.fill(*zip(*polygon))

if(__name__ == '__main__'):
    graphdataset, pos = construct_graph("gd.gv")
    partition = node_clustering(graphdataset)
    draw_vor(pos, partition)