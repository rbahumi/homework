import tensorflow as tf


class SupervisedModelPolicy():
    """
    A policy class that is wraps a pre trained Keras model for a specific gym environment
    """
    def __init__(self, env_name, model_filename):
        self._env_name = env_name
        self._model = tf.keras.models.load_model(model_filename)

    def act(self, ob):
        action = self._model.predict(ob.reshape(1, ob.shape[0]))
        return action[0]

    @property
    def model(self):
        return self._model


