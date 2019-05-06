import os
import pickle
import matplotlib.pyplot as plt


class KerasTrainStats():
    """
    Helper class that defines the training history object for each model

    The class pickles and saves the history object to disk
    """

    def __init__(self, model_name, history_dir):
        # Local import of a Keras package
        from keras.callbacks import LambdaCallback

        # Try loading history if the file exists
        history = self.load_model_history(model_name, history_dir)

        # Else, create an empty history dict
        if not history:
            history = {'loss': [],  # training loss at the end of each epoch of training
                       'val_loss': []  # validation (eval) loss at the end of each epoch of training
                       }

        self.history = history
        self.history_dir = history_dir
        self.model_name = model_name

        # Define the LambdaCallback
        self.print_callback = LambdaCallback(on_epoch_end=self.on_epoch_end)

    def on_epoch_end(self, epoch, logs):
        # Function invoked at end of each epoch.
        # It appends the 'loss', 'val_loss' and any other key found in logs to the history and save to a file
        # TODO while training in a different window a notebook to see a plot 'loss' and 'val_loss' vs. number of epochs. look at 171219-plot for an example

        for k, v in logs.items():
            self.history[k] = self.history.get(k, []) + [v]

        history_filename = '%s/history.%s.pkl' % (self.history_dir, self.model_name)
        with open(history_filename, 'wb') as fp:
            pickle.dump(self.history, fp, -1)

    @staticmethod
    def load_model_history(model_name, history_dir):
        history_filename = '%s/history.%s.pkl' % (history_dir, model_name)

        if not os.path.isfile(history_filename):
            return False

        with open(history_filename, 'rb') as f:
            return pickle.load(f)

    def plt_history(self, start_epoch=0, figsize=(20, 10), model_name=None, history_dir=None, title=None,
                    metric_str='loss'):
        """Plot the train/validation loss values"""
        if history_dir is None:
            history_dir = self.history_dir

        if model_name is None:
            # Try to load the history from a file before using self.history. This is ment for the usecase that a model
            # is trained in a different process/notebook and we want to present the latest history.
            history = self.load_model_history(self.model_name, history_dir=history_dir) or self.history
        else:
            # If model_name is given, load the history from a file
            history = self.load_model_history(model_name, history_dir=history_dir)
            if not history:
                raise Exception("Cannot find history object for model_name={model_name}".format(model_name))

        plt.figure(figsize=figsize)
        _loss = history[metric_str][start_epoch:]
        _val_loss = history['val_%s' % metric_str][start_epoch:]

        train_loss_plot, = plt.plot(range(start_epoch + 1, start_epoch + len(_loss) + 1), _loss,
                                    label='Train %s' % metric_str)
        val_loss_plot, = plt.plot(range(start_epoch + 1, start_epoch + len(_val_loss) + 1), _val_loss,
                                  label='Validation %s' % metric_str)

        plt.legend(handles=[train_loss_plot, val_loss_plot])
        plt.xlabel("epoch")
        plt.ylabel(metric_str)

        if title is not None:
            _ = plt.title(title)
