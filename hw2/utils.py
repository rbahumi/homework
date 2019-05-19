import numpy as np



def reward_to_go(episode_rewards, gamma):
    """
    Helper function that calculates the "Reward to go"

    >>> episode_rewards = np.array([1, 2, 4])
    >>> gamma = 0.5
    >>> reward_to_go(episode_rewards, gamma)
    array([3., 4., 4.])

    >>> episode_rewards = np.array([1, 2, 4])
    >>> gamma = 0.
    >>> reward_to_go(episode_rewards, gamma)
    array([1., 2., 4.])

    >>> episode_rewards = np.array([1, 2, 4])
    >>> gamma = 1.
    >>> reward_to_go(episode_rewards, gamma)
    array([7., 6., 4.])

    :param episode_rewards: np.array
    :param gamma: float
    :return: np.array
    """
    episode_reduced = []
    prev = 0.

    # Go over the rewards in reverse order
    for i in np.flip(episode_rewards):
        # The reward to go in timestamp t, is the immediate reward + discounted reward to go
        current = i + gamma * prev
        episode_reduced.append(current)
        prev = current
    # Flip the array again
    episode_reduced = np.flip(episode_reduced)
    return episode_reduced


def sum_of_rewards(episode_rewards, gamma):
    """
    Helper function that calculates the reduced sum of rewards

    >>> episode_rewards = np.array([1, 2, 4])
    >>> gamma = 0.5
    >>> sum_of_rewards(episode_rewards, gamma)
    array([3., 3., 3.])

    >>> episode_rewards = np.array([1, 2, 4])
    >>> gamma = 0.
    >>> sum_of_rewards(episode_rewards, gamma)
    array([1., 1., 1.])

    >>> episode_rewards = np.array([1, 2, 4])
    >>> gamma = 1.
    >>> sum_of_rewards(episode_rewards, gamma)
    array([7., 7., 7.])

    :param episode_rewards: np.array
    :param gamma: float
    :return: np.array
    """

    # Each element in the gammas_exp_t array is gamma^t were t is the array's index
    gammas_exp_t = gamma ** np.arange(len(episode_rewards))
    # Calculate the discounted reward
    discounted_reward = np.sum(episode_rewards * gammas_exp_t)
    # The discounted reward is used fot all the timestamps in the trajectory
    episode_reduced = discounted_reward * np.ones(shape=episode_rewards.shape)
    return episode_reduced


def normalize_array(X, eps=1e-07, axis=None):
    """
    Normalized a given np.array to have 0 mean and 1 std.

    If axis=0, the returned array will have 0 mean and 1 std along the 0 axis. (Note: constant values along the axis will have 0 mean and 0 std.)

    >>> X = np.array([[1, 2, 3], [3, 2, 1], [3, 2, 1]])
    >>> normalize_array(X, axis=None)
    array([[-1.22474472,  0.        ,  1.22474472],
           [ 1.22474472,  0.        , -1.22474472],
           [ 1.22474472,  0.        , -1.22474472]])

    >>> X = np.array([[1], [3], [5]])
    >>> normalize_array(X, axis=0)
    array([[-1.2247448],
           [ 0.       ],
           [ 1.2247448]])

    >>> X = np.array([[1, 3, 2], [3, 3, 2], [5, 3, 2]])
    >>> normalize_array(X, axis=0)
    array([[-1.2247448,  0.       ,  0.       ],
           [ 0.       ,  0.       ,  0.       ],
           [ 1.2247448,  0.       ,  0.       ]])

    >>> X = np.array([1, 2, 3])
    >>> normalize_array(X, axis=0)
    array([-1.22474472,  0.        ,  1.22474472])

    >>> X = np.array([[1, 2, 3], [3, 2, 1], [3, 2, 1]])
    >>> normalize_array(X, axis=1)
    Traceback (most recent call last):
    Exception: normalize_array(): allowed axis values are either None or 0

    :param X: np.array
    :param eps: float
    :param axis: None or 0 (if None, the array will be normalized globally, if 0 (int) the array will be normalized along the 0 axis)
    :return: np.array (X with 0 mean and 1 std)
    """
    if axis is not None and axis !=0:
        raise Exception("normalize_array(): allowed axis values are either None or 0")

    mean = np.mean(X, axis=axis)
    std = np.std(X, axis=axis) + eps

    if len(X.shape) > 1:
        # Tile 'mean' and 'std' along the 0 axis
        mean = np.tile(mean, (X.shape[0], 1))
        std = np.tile(std, (X.shape[0], 1))

    X_norm = (X - mean) / std
    return X_norm


if __name__ == "__main__":
    import doctest
    doctest.testmod()
