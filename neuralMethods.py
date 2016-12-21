import mxnet as mx
import logging

def net_constructor(hidden, activationFunction):
    """
    Defines the structure of the neural network
    :param hidden: array with the elements of each hidden layer (including the output)
    :param activationFunction: string with the activation function used in the hidden layers
    :return: object mxnet.symbol.Symbol returned with the structure of the net (the output activation function is not included)
    """
    ln = len(hidden)

    #Defining input data
    prev = mx.sym.Variable('data')
    prev = mx.sym.Flatten(data=prev)

    for i in range(ln-1):
        fc = mx.sym.FullyConnected(data = prev, name = "fc"+str(i+1), num_hidden = hidden[i])
        prev = mx.sym.Activation(data = fc, name = activationFunction+str(i+1), act_type = activationFunction)

    fc = mx.sym.FullyConnected(data = prev, name = "fc"+str(i+2), num_hidden = hidden[ln-1])

    return fc

def net_fit(mlp, epochs, alpha, train_iter, val_iter, batch_size, output_progress , ctx):
    """
    Model fitting
    :param mlp: symbol structure with the network architecture
    :param epochs: training epochs
    :param alpha: learning rate
    :param train_iter: training iterator
    :param val_iter: validation iterator
    :param batch_size: size of the batch
    :param output_progress: callback invoked on the end of the batch for printing
    :param ctx: cpu or gpu processing
    :return: fitted model
    """
    logging.getLogger().setLevel(logging.DEBUG)

    model = mx.model.FeedForward( symbol=mlp, num_epoch=epochs, learning_rate=alpha, ctx = ctx )
    model.fit( X=train_iter, eval_data=val_iter, batch_end_callback=mx.callback.Speedometer(batch_size, output_progress ) )
    return model

def predict_neural(test_iter, model):
    """
    uses a fitted model to predict the output. Computes the value of the highest coordinate of each line
    :param test_iter: test interator
    :param model: fitted model
    :return: predicted values
    """
    ypred = model.predict(test_iter)
    predictions = ypred.argmax(axis=1)
    return predictions