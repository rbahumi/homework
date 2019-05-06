import gym
import roboschool  # This import is the required for loading the roboschool environments into gym
from run_expert import run_policy
from supervised_model_policy import SupervisedModelPolicy


def run_supervised_model_policy(num_rollouts, envname, model_filename, max_timesteps=None, render=False, verbose=True):
    """
    This method is the equivalent to hw1.run_expert.run_expert_policy() for trained models

    :param num_rollouts: int
    :param envname: str (number of trajectories to generate)
    :param model_filename: str (full path to the saved model file)
    :param max_timesteps: int (maximum number of steps to generate in each trajectory)
    :param render: boolean (if true, render the environment)
    :param verbose: boolean
    :return: dict
    """
    env = gym.make(envname)
    policy = SupervisedModelPolicy(envname, model_filename)
    description = "Supervised model policy for module %s" % envname
    return run_policy(env=env, policy=policy, num_rollouts=num_rollouts, description=description,
                      max_timesteps=max_timesteps, render=render, verbose=verbose)


if __name__ == '__main__':
    # Exapmle usage:
    # python run_supervised.py 'RoboschoolHumanoid-v1' --render --num_rollouts 3

    import argparse
    from consts import ENV_TO_MODEL
    
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--model_filename', default=None)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    model_filename = args.model_filename
    if model_filename is None:
        assert args.envname in ENV_TO_MODEL
        model_filename = ENV_TO_MODEL[args.envname]

    run_supervised_model_policy(num_rollouts=args.num_rollouts, envname=args.envname, model_filename=model_filename,
                                max_timesteps=args.max_timesteps, render=args.render, verbose=True)
