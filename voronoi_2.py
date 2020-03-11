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
        partition = community.best_partition(graph)

    return partition

def merge_vor(ridge_points, ridge_vertices, regions, point_region, com_points):
    # record region indices in regions needed to be removed from vor.point_region
    region_indices = []
    for p in com_points:
        region_indices.append(point_region[int(p)])
        
    # record regions needed to be removed from vor.regions
    com_regions = []
    for i in region_indices:
        com_regions.append(regions[i])

    # record ridge in vor.ridge_vertices that needs to be removed
    vertices_pair = []
    for i in range(0, len(com_regions)-1):
        for j in range(i+1, len(com_regions)):
            pair = []
            for v in com_regions[j]:
                if(v in com_regions[i]): pair.append(v)
            if(len(pair) == 0): continue
            assert len(pair) == 2, [pair, com_regions[i], com_regions[j]]
            vertices_pair.append(pair)

    # move all vertices to the first region according to region_indices
    f = com_regions[0]
    for r in com_regions[1:]:
        for v in r:
            if(v not in f): f.append(v)

    # record ridge in vor.ridge_points that needs to be removed
    points_pair = []
    for pair in ridge_points:
        if(str(pair[0]) in ridge_points and str(pair[1]) in ridge_points):
            points_pair.append(pair)

    return com_regions[1:], region_indices[1:], points_pair, vertices_pair

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

    # merge voronoi cells for each community
    # record attributes of vor needed to be modified
    regions_remove = []
    point_region_remove = []
    ridge_points_remove = []
    ridge_vertices_remove = []
    vor_ridge_points = vor.ridge_points.tolist()
    vor_point_region = vor.point_region.tolist()
    for p_lis in com_node.values():
        regions_lis, point_region_lis, ridge_points_lis, ridge_vertices_lis\
             = merge_vor(vor_ridge_points, vor.ridge_vertices, vor.regions, vor_point_region, p_lis)
        regions_remove += regions_lis
        point_region_remove += point_region_lis
        ridge_points_remove += ridge_points_lis
        ridge_vertices_remove += ridge_vertices_lis

    # modified attributes of vor
    # vor.point_region
    for r in point_region_remove:
        vor_point_region.remove(r)
    vor.point_region = np.array(vor_point_region)

    # vor.regions
    for i in regions_remove:
        vor.regions.remove(i)
    
    # vor.ridge_points
    for pair in ridge_points_remove:
        vor_ridge_points.remove(pair)
    vor.ridge_points = np.array(vor_ridge_points)

    # vor.ridge_vertices
    for pair in ridge_vertices_remove:
        try: vor.ridge_vertices.remove(pair)
        except:
            v1, v2 = pair
            vor.ridge_vertices.remove([v2, v1])

    # plot Voronoi diagram
    voronoi_plot_2d(vor, show_vertices=False)
    plt.show()
    
if(__name__ == '__main__'):
    graphdataset, pos = construct_graph("gd.gv")
    partition = node_clustering(graphdataset)
    draw_vor(pos, partition)