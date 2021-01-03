import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch


def fig_3():

    def make_legend_arrow(legend, orig_handle,
                          xdescent, ydescent,
                          width, height, fontsize):
        p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
        return p

    prey_pos = [3,3]
    hunter_1_pos = [4,1]
    hunter_2_pos = [0,4]

    data = np.zeros([7, 7])
    fig, ax = plt.subplots(figsize=(7, 7))
    data[prey_pos[1]    , prey_pos[0]    ] = 2
    data[hunter_1_pos[1], hunter_1_pos[0]] = 1
    data[hunter_2_pos[1], hunter_2_pos[0]] = 1

    im = ax.imshow(data)
    colors = [ im.cmap(im.norm(1)),  im.cmap(im.norm(2))]
    labels = ["Hunter", "Prey"]
    patches = [ mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels)) ]

    # prey arrows
    arrow = plt.arrow(2.5,  3.0, -0.5,  0.0, color=(0,0,0), width=0.02, length_includes_head=True, head_width=(4*0.02), label='Possible Moves' )
    plt.arrow(2.5,  3.0, -0.5,  0.0, color=(1,1,1), width=0.02, length_includes_head=True, head_width=(4*0.02))
    plt.arrow(3.5,  3.0,  0.5,  0.0, color=(1,1,1), width=0.02, length_includes_head=True, head_width=(4*0.02))
    plt.arrow(3.0,  2.5,  0.0, -0.5, color=(1,1,1), width=0.02, length_includes_head=True, head_width=(4*0.02))
    plt.arrow(3.0,  3.5,  0.0,  0.5, color=(1,1,1), width=0.02, length_includes_head=True, head_width=(4*0.02))

    # hunter_1 arrows
    plt.arrow(3.5,  1.0, -0.5,  0.0, color=(1,1,1), width=0.02, length_includes_head=True, head_width=(4*0.02))
    plt.arrow(4.5,  1.0,  0.5,  0.0, color=(1,1,1), width=0.02, length_includes_head=True, head_width=(4*0.02))
    plt.arrow(4.0,  0.5,  0.0, -0.5, color=(1,1,1), width=0.02, length_includes_head=True, head_width=(4*0.02))
    plt.arrow(4.0,  1.5,  0.0,  0.5, color=(1,1,1), width=0.02, length_includes_head=True, head_width=(4*0.02))

    # hunter_2 arrows
    plt.arrow(0.0,  3.5,  0.0, -0.5, color=(1,1,1), width=0.02, length_includes_head=True, head_width=(4*0.02))
    plt.arrow(0.0,  4.5,  0.0,  0.5, color=(1,1,1), width=0.02, length_includes_head=True, head_width=(4*0.02))
    plt.arrow(0.5,  4.0,  0.5,  0.0, color=(1,1,1), width=0.02, length_includes_head=True, head_width=(4*0.02))
    plt.arrow(6.5,  4.0, -0.5,  0.0, color=(1,1,1), width=0.02, length_includes_head=True, head_width=(4*0.02))

    patches.append(arrow)
    plt.legend(handles=patches, loc=9, bbox_to_anchor=(0.5, -0.06), borderaxespad=0. , ncol=3, handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow),}),
    plt.show()



def fig_4():
    fig, (ax1, ax2) = plt.subplots(ncols=2)

    prey_pos = [3,5]
    hunter_1_pos = [2,5]
    hunter_2_pos = [4,5]

    data = np.zeros([7, 7])
    ax1.plot(figsize=(7, 7))
    data[prey_pos[1]    , prey_pos[0]    ] = 2
    data[hunter_1_pos[1], hunter_1_pos[0]] = 1
    data[hunter_2_pos[1], hunter_2_pos[0]] = 1


    im1 = ax1.imshow(data)

    prey_pos = [3,2]
    hunter_1_pos = [3,1]
    hunter_2_pos = [3,3]

    data = np.zeros([7, 7])
    ax1.plot(figsize=(7, 7))
    data[prey_pos[1]    , prey_pos[0]    ] = 2
    data[hunter_1_pos[1], hunter_1_pos[0]] = 1
    data[hunter_2_pos[1], hunter_2_pos[0]] = 1


    im2 = ax2.imshow(data)

    plt.show()
fig_3()
# fig_4()
