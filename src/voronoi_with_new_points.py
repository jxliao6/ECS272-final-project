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
from random import uniform

def construct_graph(fname, node_weight_name='weight', embedding='force-directed'):
    """
    read the graph from .gv file
    and construct the graph with the embedding algorithm
    ------------------------------------------------------
    input
    fname: str, file name
    embedding: str, embedding algorithm
    node_weight_name: name of node weight
    ------------------------------------------------------
    returns
    graphdataset: nx.Graph object
    pos: dict, {node(str): coordinates(np.array)}
    """
    # a traditional graph
    graphdataset = nx.Graph(nx.drawing.nx_pydot.read_dot(fname))

    #change the weights into ints if available
    for i in graphdataset.edges:
        graphdataset.edges[i]['weight']=int(re.findall(r'[0-9]+', (graphdataset.edges[i]['weight']))[0])
        
    for i in graphdataset.nodes:
        graphdataset.nodes[i][node_weight_name] = float(graphdataset.nodes[i][node_weight_name].strip("\""))

    # set coordinates for vertices using the corresponding embedding algorithm
    if(embedding == 'force-directed'):
        pos = nx.spring_layout(graphdataset, k=1/math.pow(len(graphdataset), 0.3),scale=10)

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
    partition: dict, {node(str or key of node): community(int)}
    """
    # compute the best partition into communities
    if(algorithm == 'modularity'):
        partition = community.best_partition(graph)

    return partition

def merge_vor(ridge_points, ridge_vertices, regions, point_region, com_points):
    # record region indices in regions needed to be removed from vor.point_region
    region_indices = []
    same_region = {}
    for p in com_points:
        region_indices.append(point_region[int(p)])
        same_region.setdefault(point_region[int(p)], []).append(p)

    #for r, lis in same_region.items():
    #    if(len(lis) > 1):
    #        print(r, lis)

    # record regions needed to be removed from vor.regions
    com_regions = []
    for i in region_indices:
        com_regions.append(regions[i])

    # record ridge in vor.ridge_vertices that needs to be removed
    vertices_pair = []
    for i in range(0, len(com_regions)-1):
        for j in range(i+1, len(com_regions)):
            if(com_regions[i] == com_regions[j]): continue
            pair = []
            for v in com_regions[j]:
                if(v in com_regions[i]): pair.append(v)
            if(len(pair) <= 1): continue #todo: check the situation that len(pair)==1 or more than 2 ???
            assert len(pair) == 2, [i, j, len(com_regions), pair, com_regions[i], com_regions[j]]
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
    
def add_boundary_points(position,partition_group, num_ps, length, width,graph):
    # add boundary points to every original nodes
    # merge into larger communities with these partions
    
    points_location = []
    label_c = len(position)
    for each_pos in position:
        ratio_lw = 1/(1 + np.exp(graph.nodes[each_pos]['weight']))
        bound_points_each = add_boundary_point_each(position[each_pos], num_ps, length*ratio_lw, width*ratio_lw)
        bound_labels = [ str(label_c+i) for i in range(num_ps)]
        label_c = label_c + num_ps
        partition_group.update(dict.fromkeys(bound_labels,partition_group[each_pos]))
        points_location.extend(bound_points_each)
        
    points_location_array = np.array(points_location)
        
    return points_location_array, partition_group
        

    
def add_boundary_point_each(center_point,numofp,length,width):

    length_points = int(length/(width+length)*numofp)
    width_points = numofp - length_points
    bottom_points = int(length_points/2)
    top_points = length_points - bottom_points
    left_points  = int(width_points/2)
    right_points = width_points - left_points
    #print(bottom_points,top_points,left_points,right_points)
    
    bottoms = [(i,center_point[1]-width/2) for i in np.linspace(center_point[0] - length/2,center_point[0] + length/2,bottom_points)]
    tops = [(i,center_point[1]+width/2) for i in np.linspace(center_point[0] - length/2,center_point[0] + length/2,top_points)]
    lefts = [(center_point[0]-length/2,i) for i in np.linspace(center_point[1] - width/2,center_point[1] + width/2,left_points)]
    rights = [(center_point[0]+length/2,i) for i in np.linspace(center_point[1] - width/2,center_point[1] + width/2,right_points)]
    return_points = bottoms + tops+lefts+rights
    
    return(return_points)
    
    
def gencoordinates(m, n, nodes_pos,r,num_rp):
    seen = list()

    while True:
        count = 0
        x, y = uniform(m, n), uniform(m, n)
        indicator = True
        for node_pos in nodes_pos:
            dist = [(a - b)**2 for a, b in zip(node_pos, (x,y))]
            dist = math.sqrt(sum(dist))
            if dist <= r:
                indicator = False
                count = count+1
                break
        
        if indicator == True:
            seen.append((x,y))
            
        #if len(seen)%10==0 and len(seen)> 1:
            #print(len(seen),count)
        
        if len(seen) == num_rp:
            return seen

def draw_vor(graph,pos, partition,num_ps,length,width,r,num_rp):

    # stack coordinates of nodes in coor_lis to draw the Voronoi Diagram
    # indices of coor_lis correspond to the node
    coor_lis = []
    for p in range(0, len(pos.keys())): # p in this case. #todo: more general
    #for p in pos.keys():
        coor_lis.append(pos[str(p)])
    vorpoints = np.stack(coor_lis)
    
    
        
    # add boundary points for existing points with partitions
    boundary_points, partition2  = add_boundary_points(pos,partition, num_ps, length, width,graph)
    vorpoints2 = np.concatenate((vorpoints, boundary_points))
    
    # add random long distance points
    # they are not in the partition list
    random_points = gencoordinates(-10, 10, vorpoints2,r,num_rp)
    
    vorpoints3 = np.concatenate((vorpoints, boundary_points, random_points))
    vor = Voronoi(vorpoints3)
    #vor = Voronoi(vorpoints2)
    #vor = Voronoi(vorpoints)
    """
    print(vor._ridge_dict)
    print(vor.ndim)
    print(vor.npoints)
    print(vor._points)
    print(vor.min_bound)
    print(vor.max_bound)
    return 0
    """
    voronoi_plot_2d(vor, show_vertices=False)
    plt.show()
    
    # build dictionary to store community info
    # com_node = {community: [nodes in community]}
    com_node = {}
    for p, com in partition2.items():
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
    draw_vor(graphdataset, pos, partition, 40, 0.1,0.1, 0.2, 100)
