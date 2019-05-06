import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def create_dir_if_not_exists(name):
    if not os.path.exists(name):
        os.makedirs(name)


def grid_iterator(n_rows, n_cols, axis_size=2.5, total=None, fig_title=None):
    """
    matplotlib helper method for plotting a grid of graphs


    :param n_rows: int
    :param n_cols: int
    :param axis_size: float
    :param total: int (stopping condition - stop before n_rows * n_cols subplots are filled in the grid)
    :param fig_title: str
    """
    fig = plt.figure()
    gs = gridspec.GridSpec(n_rows, n_cols, top=1., bottom=0., right=1., left=0., hspace=0.15, wspace=0.1)

    index = 0
    for r in range(n_rows):
        for c in range(n_cols):
            ax = fig.add_subplot(gs[r, c])
            yield ax, index

            index += 1
            # Allowing for not filling all spots
            if total and index >= total:
                break

    width, height = n_cols * axis_size, n_rows * axis_size
    fig.set_size_inches(width, height)

    # Add title to the figure
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=16, y=1.05)
