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


#from matplotlib.patches import Polygon




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
    partition: dict, {node(str): community(int)}
    """
    # compute the best partition into communities
    if(algorithm == 'modularity'):
        partition = community.best_partition(graph)

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

    #colorRegions(vor)
    
    voronoi_plot_2d(vor, show_vertices=False)
    plt.show()
    return vor
    
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
        
        # if len(seen)%10==0 and len(seen)> 1:
        #     print(len(seen),count)
    
        if len(seen) == num_rp:
            return seen

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

def isSameRegion(r1, r2):
    if len(r1) != len(r2):
        return False
    for i in range(len(r1)):
        if r1[i] != r2[i]:
            return False
    return True

def colorRegions(vor):
    #ignore virtual regions
    real_regions = vor.regions.copy()
    virtual_regions = []
    for i in range(500):
        virtual_regions.append(vor.regions[vor.point_region[len(vor.point_region) - i - 1]])

    for vregion in virtual_regions:
        for rregion in real_regions:
            if isSameRegion(vregion, rregion):
                real_regions.remove(rregion)
                break
                        

    region_id = {} #dict{key: id, value: list of vertices(==region)}
    idcount = 0
    for region in real_regions:
        region_id[idcount] = region
        idcount += 1

    #First create a stack containing list

    neighbors = {} # dict{key: id, value: list of neighbors}
    for id in region_id: # find the neighbors for each region
        neighbors[id] = [] 
        for id2 in region_id:
            if id == id2:
                continue
            hasEdge = 0
            for element in region_id[id2]:
                if element in region_id[id]:
                    hasEdge += 1
            if hasEdge >= 2:
                neighbors[id].append(id2)

    # for neighbor in neighbors:
    #     print(str(neighbor) + ": " + str(neighbors[neighbor]))
    neighbors_copy = neighbors.copy()

    id_stack = [] #coloring order
    replace_dic = {}
    #print(len(region_id))
    while len(neighbors) > 0: #calculate the order of coloring
        #tobe_removed = []
        for id in neighbors:
            if len(neighbors[id]) < 6: #First Case
                id_stack.append(id)
                #tobe_removed.append(id)
                for id2 in neighbors:
                    if id in neighbors[id2]:
                        neighbors[id2].remove(id)
                del neighbors[id]
                break
            # elif len(region_id[id]) == 5: #Second Case
            #     n1 = -1
            #     n2 = -1
            #     for id2 in region_id[id]:
            #         if len(region_id[id2]) <= 7:
            #             for id3 in region_id[id2]:
            #                 if len(region_id[id3] <= 7) and id2 not in region_id[id3]:
            #                     n1 = id2
            #                     n2 = id3
            #                     for ids in region_id: # replace n1 and n2 with (n1,n2)
            #                         if n1 in region_id[ids]:
            #                             region_id[ids].remove(n1)
            #                             region_id[ids].append(idcount)
            #                         if n2 in region_id[ids]:
            #                             region_id[ids].remove(n2)
            #                             if idcount not in region_id[ids]:
            #                                 region_id[ids].append(idcount)
            #                     replace_dic[idcount] = [id2, id3]
            #                     idcount += 1
            #                     break
            #     if n1 == -1 or n2 == -1: #debug
            #         print("n1 and n2 not found!!!")


    color_dict = {} #region-color mapping
    color_list = ['#FE2712','#FB9902','#B2D732','#347C98','#0247FE','#8601AF','#C21460']
    color_freq = {}
    for color in color_list:
        color_freq[color] = 0

    while len(id_stack) > 0: #deciding colors in each region
        temp_list = color_list.copy()
        id = id_stack.pop()
        for id2 in neighbors_copy[id]:
            if id2 in color_dict and color_dict[id2] in temp_list:
                temp_list.remove(color_dict[id2])
        
        #least used color first
        leastColor = temp_list[0]
        leastCount = color_freq[leastColor]
        for color in temp_list:
            if color_freq[color] < leastCount:
                leastColor = color
                leastCount = color_freq[color]
        color_dict[id] = leastColor
        color_freq[leastColor] = color_freq[leastColor] + 1
        
    for color in color_freq:
        print(color + ": " + str(color_freq[color]))
    
    for id in region_id: # assign color for each region
        if not -1 in region_id[id]:
            polygon = [vor.vertices[i] for i in region_id[id]]
            plt.fill(*zip(*polygon), color = color_dict[id], alpha = 0.4)



# def colorRegions2(vor):
#     regions, vertices = voronoi_finite_polygons_2d(vor)
#     polygons = []
#     for reg in regions:
#         polygon = vertices[reg]
#         polygons.append(polygon)
    
#     for poly in polygons:


if(__name__ == '__main__'):
    graphdataset, pos = construct_graph("gd.gv")

    coor_lis = []
    for p in range(0, len(pos.keys())):
        coor_lis.append(pos[str(p)])
    vorpoints = np.stack(coor_lis)
    

    size_of_random_points = 500
    random_points = gencoordinates(-10, 10, vorpoints, 0.01, size_of_random_points)
    vorpoints2 = np.concatenate((np.stack(coor_lis), random_points))
    
    vor = Voronoi(vorpoints2)
    
    colorRegions(vor)
    
    #voronoi_plot_2d(vor, show_vertices=False)
    
    plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
    plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)
    plt.show()

    #partition = node_clustering(graphdataset) 
    #draw_vor(pos, partition)
