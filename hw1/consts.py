EXPERT_DIR = "experts"
EXPERT_DATA_DIR = 'expert_data'
SUPERVISED_MODELD_DATA_DIR = 'hw1/supervised_modeled_data'
ROBOSCOOL_AVAILABLE_ENVS = ['RoboschoolAnt-v1', 'RoboschoolHumanoid-v1', 'RoboschoolHalfCheetah-v1', 'RoboschoolReacher-v1',
                  'RoboschoolHopper-v1', 'RoboschoolWalker2d-v1']

## Added the model trained in the 'Behavioral Cloning.ipynb' notebook (these model weren't added to the repository)
ENV_TO_MODEL = {
    'RoboschoolAnt-v1': 'models/model_RoboschoolAnt-v1_layers_100_3_neurons_l2_0.0001_Adam_optimizer_0.001_lr_None_dropout_without_batchnorm/base.hdf5',
    'RoboschoolHalfCheetah-v1': 'models/model_RoboschoolHalfCheetah-v1_layers_100_3_neurons_l2_0.0001_Adam_optimizer_0.001_lr_None_dropout_without_batchnorm/base.hdf5',
    'RoboschoolHopper-v1': 'models/model_RoboschoolHopper-v1_layers_100_3_neurons_l2_0.0001_Adam_optimizer_0.001_lr_None_dropout_without_batchnorm/base.hdf5',
    'RoboschoolHumanoid-v1': 'models/model_RoboschoolHumanoid-v1_layers_100_3_neurons_l2_0.0001_Adam_optimizer_0.001_lr_None_dropout_without_batchnorm/base.hdf5',
    'RoboschoolReacher-v1': 'models/model_RoboschoolReacher-v1_layers_100_3_neurons_l2_0.0001_Adam_optimizer_0.001_lr_None_dropout_without_batchnorm/base.hdf5',
    'RoboschoolWalker2d-v1': 'models/model_RoboschoolWalker2d-v1_layers_100_3_neurons_l2_0.0001_Adam_optimizer_0.001_lr_None_dropout_without_batchnorm/base.hdf5'
}