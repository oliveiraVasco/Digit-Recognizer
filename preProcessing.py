from numpy import float32

def segment_data(data, perc, train, labels):
    """
    Segments the neural network's input data (X). Does it for the training or cross validation according with the percentage.
    :param data: 2D array with labels on the first column and the rest of the data in the other
    :param perc: data percentage for training
    :param train: boolean for returning training (Frue), or cross validation (False) data.
    :param labels: boolen for returning inputs (X, True), or labels(Y, False).
    :return: segmented data on a 2D array
    """
    m = data.shape[0]
    div = int(perc * m)
    if train == True:
        if labels == True:
            dt = data[0:div, 0]
        else:
            dt = data[0:div, 1:data.shape[1]]
    else:
        if labels == True:
            dt = data[div + 1:m, 0]
        else:
            dt = data[div + 1:m, 1:data.shape[1]]
    return dt


def to4d(img):
    """
    Reshape function original from http://mxnet.io/tutorials/python/mnist.html
    :param img: Array with segmented data
    :return: data reshaped. Each image (line on input) is now represented by a 28x28 matrix
    """
    return img.reshape(img.shape[0], 1, 28, 28).astype(float32)/255


import matplotlib.pyplot
def plotNumber(image):
    """
    Image plot inspired on http://mxnet.io/tutorials/python/mnist.html
    :param image: row of all pixels with the label as first element
    :return:
    """
    image = image[1:image.shape[0]]
    image = image.reshape(28,28)
    matplotlib.pyplot.imshow(image, cmap='Greys_r')
    matplotlib.pyplot.show()
    return