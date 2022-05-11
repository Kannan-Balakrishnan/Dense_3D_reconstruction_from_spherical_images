from matplotlib import pyplot as plt
import numpy as np

def plot(cross_product1, cross_product2, coplanar_points, x_prime_vector, x_prime_vector_scaled, i):
    fig = plt.figure()
    ax = plt.axes(projection = "3d")
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    start = np.array([0, 0, 0])
    #coplanar_points = -coplanar_points
    ax.quiver(start[0],start[1],start[2],x_prime_vector[0], x_prime_vector[1], x_prime_vector[2], color = "r", label = "x_min")
    ax.quiver(start[0],start[1],start[2],x_prime_vector_scaled[0], x_prime_vector_scaled[1], x_prime_vector_scaled[2], color = "g", label = "x_max")
    ax.quiver(start[0],start[1],start[2], coplanar_points[i][0], coplanar_points[i][1], coplanar_points[i][2])
    ax.quiver(start[0],start[1],start[2], cross_product1[0], cross_product1[1], cross_product1[2], color = "yellow")
    ax.quiver(start[0],start[1],start[2], cross_product2[0], cross_product2[1], cross_product2[2], color = "orange")
    plt.show()