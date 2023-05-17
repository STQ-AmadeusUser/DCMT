import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

def grids():
    sz = 31
    stride = 8

    sz_x = sz // 2
    sz_y = sz // 2

    x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                       np.arange(0, sz) - np.floor(float(sz_y)))

    grid_to_search_x = x * stride + 255 // 2
    grid_to_search_y = y * stride + 255 // 2

    return grid_to_search_x, grid_to_search_y

if __name__ == '__main__':

    xs, ys = grids()

    fig, ax = plt.subplots(1)
    boxes = [Rectangle((0.0, 0.0), 255.0, 255.0)]
    pc = PatchCollection(boxes, facecolor='black', edgecolor='none', alpha=0.9)
    ax.add_collection(pc)
    # _ = ax.errorbar(0.0, 0.0, 255.0, 255.0, fmt='none', ecolor='k')
    plt.scatter(xs, ys, c='yellow', s=4)
    # rect = Rectangle(
    #     (box[0][0], box[0][1]), box[0][2] - box[0][0], box[0][3] - box[0][1],
    #     linewidth=1,
    #     edgecolor='r',
    #     facecolor='none'
    # )
    plt.show()


