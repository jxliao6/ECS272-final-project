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
from plotly.tools import mpl_to_plotly

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
    graphdataset  = nx.relabel.convert_node_labels_to_integers(graphdataset)

    #change the weights into ints
    if fname == "gd.gv":
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
            
def gen_canvas_bound_coordiates(lim_s,lim_l):
    point_range = np.linspace(lim_s,lim_l,50)
    left = [(lim_s,i)  for i in point_range]
    right = [(lim_l,i)  for i in point_range]
    top = [(i,lim_l)  for i in point_range]
    bottom = [(i,lim_s)  for i in point_range]
            
    return np.array(left+right+top+bottom)

def draw_vor(pos, partition):

    # stack coordinates of nodes in coor_lis to draw the Voronoi Diagram
    # indices of coor_lis correspond to the node
    coor_lis = []
    for p in range(0, len(pos.keys())):
        coor_lis.append(pos[p])
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

def colorRegions(vor,random_points_num, partition):
    #ignore virtual regions
    real_regions = vor.regions.copy()
    virtual_regions = []
    for i in range(random_points_num):
        virtual_regions.append(vor.regions[vor.point_region[len(vor.point_region) - i - 1]])

    for vregion in virtual_regions:
        for rregion in real_regions:
            if isSameRegion(vregion, rregion):
                real_regions.remove(rregion)
                break
                        

    region_id = {} #{key: point id, value: list of vertices(==region)}
    #print(len(vor.points) - random_points_num)
    for p in range(len(vor.points) - random_points_num):
        for r in real_regions:
            if isSameRegion(vor.regions[vor.point_region[p]], r):
                region_id[p] = r
                break
    

    ##################################################################
    # region_id = {} #dict{key: id, value: list of vertices(==region)}
    # idcount = 0
    # for region in real_regions:
    #     region_id[idcount] = region
    #     idcount += 1
    ##################################################################
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

    cluster_id = {} #{key: cluster id, value: list of region id}
    for r in region_id:
        
        if partition[r] not in cluster_id:
            cluster_id[partition[r]] = []
        elif r in cluster_id[partition[r]]:
            continue
        cluster_id[partition[r]].append(r)

    cluster_neighbor = {} #{key: cluster id, value: list of region's neighbor id}
    for c in cluster_id:
        cluster_neighbor[c] = []
        for r in cluster_id[c]:
            for n in neighbors[r]:
                if partition[n] not in cluster_neighbor[c]:
                    cluster_neighbor[c].append(partition[n])

    # for c in cluster_neighbor:
    #     print(cluster_neighbor[c])
    # for neighbor in neighbors:
    #     print(str(neighbor) + ": " + str(neighbors[neighbor]))
    neighbors_copy = cluster_neighbor.copy()

    cluster_stack = [] #coloring order
    replace_dic = {}
    #print(len(region_id))
    while len(cluster_neighbor) > 0: #calculate the order of coloring
        #tobe_removed = []
        print(len(cluster_neighbor))
        for id in cluster_neighbor:
            print(len(cluster_neighbor[id]))
            if len(cluster_neighbor[id]) < 12: #First Case
                cluster_stack.append(id)
                #tobe_removed.append(id)
                for id2 in cluster_neighbor:
                    if id in cluster_neighbor[id2]:
                        cluster_neighbor[id2].remove(id)
                del cluster_neighbor[id]
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

    print("here")
    color_dict = {} #region-color mapping
    color_list = ['#FE2712','#FC600A','#FB9902','#FCCC1A','#FEFE33','#B2D732','#66B032','#347C98','#0247FE','#4424D6','#8601AF','#C21460']
    color_freq = {}
    for color in color_list:
        color_freq[color] = 0

    # while len(id_stack) > 0: #deciding colors in each region
    #     temp_list = color_list.copy()
    #     id = id_stack.pop()
    #     for id2 in neighbors_copy[id]:
    #         if id2 in color_dict and color_dict[id2] in temp_list:
    #             temp_list.remove(color_dict[id2])
        
    #     #least used color first
    #     leastColor = temp_list[0]
    #     leastCount = color_freq[leastColor]
    #     for color in temp_list:
    #         if color_freq[color] < leastCount:
    #             leastColor = color
    #             leastCount = color_freq[color]
    #     color_dict[id] = leastColor
    #     color_freq[leastColor] = color_freq[leastColor] + 1
    
    while len(cluster_stack) > 0: #deciding colors in each region
        temp_list = color_list.copy()
        id = cluster_stack.pop()
        # if(id in color_dict): # skip if we have already colored this region
        #     continue
        for id2 in neighbors_copy[id]:
            if id2 in color_dict and color_dict[id2] in temp_list:
                temp_list.remove(color_dict[id2])


        # print("Cluster: ")
        # print(cluster)
        # for id in cluster: # find all neighbor colors used
        #     for id2 in neighbors_copy[id]:
        #         if id2 in color_dict and color_dict[id2] in temp_list:
        #             temp_list.remove(color_dict[id2])
        
        #least used color first
        leastColor = temp_list[0]
        leastCount = color_freq[leastColor]
        for color in temp_list:
            if color_freq[color] < leastCount:
                leastColor = color
                leastCount = color_freq[color]
        color_dict[id] = leastColor
        color_freq[leastColor] = color_freq[leastColor] + len(cluster_id[id])
        
    
    for color in color_freq:
        print(color + ": " + str(color_freq[color]))
    
    for r in cluster_id:
        for id in cluster_id[r]: # assign color for each region
        # if not -1 in region_id[id]:
            polygon = [vor.vertices[i] for i in region_id[id]]
            #plt.fill(*zip(*polygon), color = color_dict[id], alpha = 0.4)
            plt.fill(*zip(*polygon), color = color_dict[r])


def add_boundary_points(position,partition_group, num_ps, length, width,graph):
    # add boundary points to every original nodes
    # merge into larger communities with these partions
    
    points_location = []
    label_c = len(position)
    for each_pos in position:
        ratio_lw = np.sqrt(float(graph.nodes[each_pos]['fontsize'].strip('"'))/12.5)
        bound_points_each = add_boundary_point_each(position[each_pos], num_ps, length*ratio_lw, width*ratio_lw)
        bound_labels = [ label_c+i for i in range(num_ps)]
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

# def colorRegions2(vor):
#     regions, vertices = voronoi_finite_polygons_2d(vor)
#     polygons = []
#     for reg in regions:
#         polygon = vertices[reg]
#         polygons.append(polygon)
    
#     for poly in polygons:


if(__name__ == '__main__'):
    ax = plt.figure(1)
    graphdataset, pos = construct_graph("univ.gv")
    
    #clustering
    partition = node_clustering(graphdataset)

    coor_lis = []
    for p in pos.keys():
        coor_lis.append(pos[p])
    vorpoints = np.stack(coor_lis)
    
    # add boundary points for existing points with partitions
    boundary_points, partition2  = add_boundary_points(pos,partition, 40, 0.2, 0.2,graphdataset)
    vorpoints2 = np.concatenate((vorpoints, boundary_points))
    
    print("finish vorpoint2!")
    
    # add random points
    size_of_random_points = 2000
    random_points = gencoordinates(-10, 10, vorpoints, 0.5, size_of_random_points)
    canvas_bound_points = gen_canvas_bound_coordiates(-11,11)
    vorpoints3 = np.concatenate((np.stack(coor_lis),boundary_points, canvas_bound_points, random_points))
    
    print("finish vorpoint3!")
    
    vor = Voronoi(vorpoints3)
    
    #print(partition)
    colorRegions(vor,size_of_random_points+len(canvas_bound_points), partition2)
    
    #voronoi_plot_2d(vor, show_vertices=False)

    #Text and edges
    for p in pos:
        plt.text(pos[p][0], pos[p][1], graphdataset.nodes[p]['label'].strip("\"").split("\\n")[0], ha='center', va='center', fontsize = 5)
    #nx.draw_networkx_edges(graphdataset, pos=pos,alpha=0.2)


    #     xlist = []
    #     for x in pos[p]:
    #         print(p)
        #ylist = [r[1] for r in pos[p]]

        # x = sum(xlist) / len(xlist)
        # y = sum(ylist) / len(ylist)
        # plt.text(x, y, graphdataset.nodes[p]['label'].split()[-1][:-1], ha='center', va='center', fontsize = 5)
        #print("X: " + str(pos[p][0]) + "Y: " + str(pos[p][1]) + graphdataset.nodes[p]['label'])
    
    plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
    plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)
    #plt.show()
    #draw_vor(pos, partition)

    plotly_fig = mpl_to_plotly(ax)
    plotly_fig.show()
