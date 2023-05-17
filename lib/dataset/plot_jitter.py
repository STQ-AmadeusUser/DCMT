import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


def generate_jitter_shift():
    add = np.array([4, 8, 12, 16]).astype(np.float32)
    minus = -1 * add
    add2 = add.reshape(4, 1).repeat(2, axis=-1)
    minus2 = minus.reshape(4, 1).repeat(2, axis=1)

    shift = np.zeros((96, 4))

    # settle (x1, y1) change (x2, y2)
    shift[0:4, 2] += add
    shift[4:8, 2] += minus
    shift[8:12, 3] += add
    shift[12:16, 3] += minus
    shift[16:20, 2:4] += add2
    shift[20:24, 2:4] += minus2

    # settle (x2, y1) change (x1, y2)
    shift[24:28, 0] += add
    shift[28:32, 0] += minus
    shift[32:36, 3] += add
    shift[36:40, 3] += minus
    shift[40:44, 0] += add
    shift[40:44, 3] += add
    shift[44:48, 0] += minus
    shift[44:48, 3] += minus

    # settle (x2, y2) change (x1, y1)
    shift[48:52, 0] += add
    shift[52:56, 0] += minus
    shift[56:60, 1] += add
    shift[60:64, 1] += minus
    shift[64:68, 0:2] += add2
    shift[68:72, 0:2] += minus2

    # settle (x1, y2) change (x2, y1)
    shift[72:76, 2] += add
    shift[76:80, 2] += minus
    shift[80:84, 1] += add
    shift[84:88, 1] += minus
    shift[88:92, 1:3] += add2
    shift[92:96, 1:3] += minus2

    return shift


if __name__ == '__main__':
    box = np.array([60.0, 60.0, 120.0, 120.0]).reshape(1, 4)
    box_rep = box.repeat(96, axis=0)
    shift = generate_jitter_shift()
    shift = np.concatenate([
        shift[..., 0, None],
        -shift[..., 3, None],
        shift[..., 2, None],
        -shift[..., 1, None]
    ], 1)
    jitter_box = box_rep + shift
    x1 = jitter_box[..., 0]
    y1 = jitter_box[..., 1]
    x2 = jitter_box[..., 2]
    y2 = jitter_box[..., 3]
    w = x2 - x1
    h = y2 - y1

    fig, ax = plt.subplots(1)
    boxes = [
        Rectangle((x_, y_), w_, h_) for x_, y_, w_, h_ in zip(x1[24:48], y1[24:48], w[24:48], h[24:48])
    ]
    # boxes = [
    #     Rectangle((60.0, 60.0), 60.0, 60.0)
    # ]
    pc = PatchCollection(boxes)
    # pc = PathCollection(boxes, facecolor='r', edgecolor='none', alpha=0.5)
    ax.add_collection(pc)
    _ = ax.errorbar(x1[24:48], y1[24:48], xerr=w[24:48], yerr=h[24:48], fmt='none', ecolor='k')
    # rect = Rectangle(
    #     (box[0][0], box[0][1]), box[0][2] - box[0][0], box[0][3] - box[0][1],
    #     linewidth=1,
    #     edgecolor='r',
    #     facecolor='none'
    # )
    plt.show()


