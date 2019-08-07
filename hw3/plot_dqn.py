import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dqn import HDF_KEY


def grid_iterator(n, n_cols=4, axis_size=2.5, fig_title=None, return_index=True, top=1., bottom=0., right=1., left=0.,
                  hspace=0.15, wspace=0.1, fig_title_fontsize=16, fig_title_y=1.05):
    """
    matplotlib helper method for plotting a grid of graphs

    :param n: int
    :param n_cols: int
    :param axis_size: float
    :param fig_title: str
    :param return_index: boolean
    """
    fig = plt.figure()
    n_rows = math.ceil(n / n_cols)
    gs = gridspec.GridSpec(n_rows, n_cols, top=top, bottom=bottom, right=right, left=left, hspace=hspace, wspace=wspace)

    # Add title to the figure
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=fig_title_fontsize, y=fig_title_y)

    # If there is only one row, the number of cols should be exactly n
    if n_rows == 1:
        n_cols = n

    width, height = n_cols * axis_size, n_rows * axis_size
    fig.set_size_inches(width, height)

    index = 0
    for r in range(n_rows):
        for c in range(n_cols):
            ax = fig.add_subplot(gs[r, c])
            yield ax, index if return_index else ax

            index += 1
            # Stop at the n'th plot
            if index >= n:
                break


def load_results_df(filename):
    return pd.read_hdf(filename, key=HDF_KEY)


def plot_results_per_metric(title_to_hdf, n_cols, fig_title=None, axis_size=4.5):
    plt_grid = grid_iterator(n=2, n_cols=2, axis_size=axis_size, fig_title=fig_title, return_index=True, wspace=0.3, fig_title_y=1.2)

    for metric in ['mean_episode_reward', 'best_mean_episode_reward']:
        ax, i = plt_grid.__next__()

        for title, df_filename in title_to_hdf.items():
            df = load_results_df(df_filename)
            ax.plot(df['timestep'].values / 1000, df[metric].values, label=title)

        ax.set_title(metric.replace("_", " ").title())
        ax.set_aspect('auto')
        ax.set_xlabel("timesteps (K)")
        if not i % n_cols:
            ax.set_ylabel("reward")
        ax.legend()


def plot_results_graph_per_title(title_to_hdf, n_cols, fig_title=None, axis_size=4.5):
    """
    Plots a single graph for each title

    :param title_to_hdf:
    :param n_cols:
    :param fig_title:
    :param axis_size:
    :return:
    """
    plt_grid = grid_iterator(n=len(title_to_hdf), n_cols=n_cols, axis_size=axis_size, fig_title=fig_title, return_index=True, wspace=0.3, fig_title_y=1.2)

    for title, df_filename in title_to_hdf.items():
        ax, i = plt_grid.__next__()
        df = load_results_df(df_filename)

        ax.plot(df['timestep'].values / 1000, df['mean_episode_reward'].values, label="mean_episode_reward")
        ax.plot(df['timestep'].values / 1000, df['best_mean_episode_reward'].values, label="best_mean_episode_reward")
        ax.set_title(title)
        ax.set_aspect('auto')
        ax.set_xlabel("timesteps (K)")
        if not i % n_cols:
            ax.set_ylabel("reward")
        ax.legend()