"""
Helper module that handles Keras models
"""
import keras
from sklearn.metrics import mean_squared_error


def create_model(input_dim, output_dim, num_hidden=3, units=100, l2_reg=None,
                 dropout=None, use_batchnorm=True, **kwargs):
    """
    This method wraps the creation of a Keras fully connected model

    :param input_dim: int (the number of the model's input units)
    :param output_dim: int (the number of the model's output units)
    :param num_hidden: int (number of hidden layer)
    :param units: int (number of units in each hidden layer)
    :param l2_reg: None or float (amount of kernel regularization in each layer, float in [0.0, 1.0])
    :param dropout: None or float (amount of dropout rate in each layer, float in [0.0, 1.0])
    :param use_batchnorm: Boolean (if True, add BatchNormalization layer after the input layer and each of the hidden layers)
    :return: keras.models.Model instance
    """
    if l2_reg is None:
        l2_reg = 0.

    input_layer = keras.layers.Input(shape=(input_dim,), name="input")

    # Define model architecture
    if use_batchnorm:
        layer = keras.layers.BatchNormalization()(input_layer)
    else:
        layer = input_layer

    if dropout is not None:
        # Add dropout layer
        layer = keras.layers.Dropout(dropout)(layer)

    for i in range(1, num_hidden + 1):
        layer = keras.layers.Dense(units=units, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_reg))(layer)
        if use_batchnorm:
            layer = keras.layers.BatchNormalization()(layer)

        # Add dropout layer
        if dropout is not None:
            layer = keras.layers.Dropout(dropout)(layer)

    # Add a prediction layer
    predictions = keras.layers.Dense(output_dim, activation=None)(layer)

    # Note using the input object 'page_id_mapping_input' not 'page_id_embedings'
    model = keras.models.Model(inputs=[input_layer], outputs=predictions)
    return model



def get_compiled_model(input_dim, output_dim, num_hidden=3, units=100, l2_reg=1e-04, lr=1e-03, dropout=None,
                       use_batchnorm=False, optimizer_cls=keras.optimizers.Adam, **kwargs):
    """
    Wraps the creation of a compiled Keras fully connected model

    :param input_dim: int (the number of the model's input units)
    :param output_dim: int (the number of the model's output units)
    :param num_hidden: int (number of hidden layer)
    :param units: int (number of units in each hidden layer)
    :param l2_reg: None or float (amount of kernel regularization in each layer, float in [0.0, 1.0])
    :param lr: float (learning rate, float in [0.0, 1.0])
    :param dropout: None or float (amount of dropout rate in each layer, float in [0.0, 1.0])
    :param use_batchnorm: Boolean (if True, add BatchNormalization layer after the input layer and each of the hidden layers)
    :param optimizer_cls: class object (keras.optimizers.Optimizer subclass)
    :return: compiled keras.models.Model instance
    """

    # Create a model
    model = create_model(input_dim=input_dim, output_dim=output_dim, num_hidden=num_hidden, units=units,
                         l2_reg=l2_reg, dropout=dropout, use_batchnorm=use_batchnorm, **kwargs)

    # Compile the model
    optimizer = optimizer_cls(lr=lr)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

    return model


def get_model_name(base_name, num_hidden=3, units=100, l2_reg=0., optimizer_cls=keras.optimizers.Adam, lr=1e-03, dropout=None, use_batchnorm=False, **kwargs):
    """
    Generates a model name based on the 'base_name' and the given networks parameters

    :param base_name: str
    :param num_hidden: int (number of hidden layer)
    :param units: int (number of units in each hidden layer)
    :param l2_reg: float (amount of kernel regularization in each layer, float in [0.0, 1.0])
    :param optimizer_cls: class object (keras.optimizers.Optimizer subclass)
    :param lr: float (learning rate, float in [0.0, 1.0])
    :param dropout: None or float (amount of dropout rate in each layer, float in [0.0, 1.0])
    :param use_batchnorm: Boolean (if True, add BatchNormalization layer after the input layer and each of the hidden layers)
    :return: str
    """
    batchnorm = "with" if use_batchnorm else "without"
    optimizer = optimizer_cls.__name__
    model_name = "model_{base_name}_layers_{units}_{num_hidden}_neurons_l2_{l2_reg}_{optimizer}_optimizer_{lr}_lr_{dropout}_dropout_{batchnorm}_batchnorm"\
        .format(base_name=base_name, units=units, num_hidden=num_hidden, l2_reg=l2_reg, optimizer=optimizer, lr=lr, dropout=dropout, batchnorm=batchnorm)
    return model_name


def calc_mse(model, dataset_name, X_train, X_test, y_train, y_test):
    """Helper function that calculates the MSE for each of the train/validation/test sets"""
    y_pred_train = model.predict([X_train])
    train_mse = mean_squared_error(y_pred_train, y_train)

    y_pred_test = model.predict([X_test])
    test_mse = mean_squared_error(y_pred_test, y_test)

    return dict(dataset_name=dataset_name, train_mse=train_mse, test_mse=test_mse)
