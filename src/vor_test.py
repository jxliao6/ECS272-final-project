from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import numpy as np
from shapely import ops
from shapely import geometry

def simple_voronoi(vor, saveas=None, lim=None):
    # Make Voronoi Diagram 
    fig = voronoi_plot_2d(vor, show_points=True, show_vertices=True, s=4)
    # Configure figure 
    fig.set_size_inches(5,5)
    plt.axis("equal")
    if lim:
        plt.xlim(*lim)
        plt.ylim(*lim)
    if not saveas is None:
        plt.savefig("../pics/%s.png"%saveas)
    plt.show()

def merge_vor(vor1, vor2):
    ax = plt.subplot(1, 1, 1)
    lines1 = [
        geometry.LineString(vor1.vertices[line])
        for line in vor1.ridge_vertices
        if -1 not in line
    ]
    lines2 = [
        geometry.LineString(vor2.vertices[line])
        for line in vor2.ridge_vertices
        if -1 not in line
    ]
    
    for coords in vor1.points:
        ax.scatter(coords[0], coords[1])

    for coords in vor2.points:
        ax.scatter(coords[0], coords[1])
    #points1 = vor1.points.tolist()
    #for x,y in points1:
    #    ax.scatter(x,y)

    #points2 = vor2.points.tolist()
    #for x,y in points2:
    #    ax.scatter(x,y)
    shape1 = ops.unary_union(list(ops.polygonize(lines1)))
    x1, y1 = shape1.exterior.xy
    ax.plot(x1,y1)

    shape2 = ops.unary_union(list(ops.polygonize(lines2)))
    x2, y2 = shape2.exterior.xy
    ax.plot(x2,y2)
    plt.show()
    

n_randoms = 10

for i in range(1):
    randos = np.random.rand(n_randoms, 2).tolist()
    vor1 = Voronoi(randos)
    simple_voronoi(vor1, lim=(-1.5,2.5), saveas=None)
    randos = np.random.rand(n_randoms, 2).tolist()
    vor2 = Voronoi(randos)
    simple_voronoi(vor2, lim=(-1.5,2.5), saveas=None)
    merge_vor(vor1, vor2)