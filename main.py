import numpy
import preProcessing as pp
import mxnet as mx
import neuralMethods as nn

def main():
    '''
    Tasks performed:
        Read data from CSV;
        Segmentation of data in train and cross-validation;
        Definition of iterators;
        Network construction;
        Model Fit;
        Predictions and file writing.
    :return:
    '''

    print ">>>>>>Start collecting"
    orgData = numpy.loadtxt(open("data/train.csv","rb"), dtype = 'int', delimiter=",",skiprows=1)
    testData = numpy.loadtxt(open("data/test.csv", "rb"), dtype='int', delimiter=",", skiprows=1)
    print ">>>>>>Data collected"

    #pp.plotNumber(orgData[0,:]) #Number Printing

    k = 10 # outputs

    # Segmentation of the datasets on training and cross-validation
    perc = 0.7 # percentage of data for training; (1-perc) for cross-validation
    trainX = pp.segment_data(orgData, perc, True, False)
    cvX = pp.segment_data(orgData, perc, False, False)
    trainY = pp.segment_data(orgData, perc, True, True)
    cvY = pp.segment_data(orgData, perc, False, True)

    # Batch size used on the fitting
    batch_size = 100
    # Iterators
    train_iter = mx.io.NDArrayIter(pp.to4d(trainX), trainY, batch_size, shuffle = True)
    val_iter = mx.io.NDArrayIter(pp.to4d(cvX), cvY, batch_size)
    test_iter = mx.io.NDArrayIter(pp.to4d(testData))

    # Construction of the net architecture
    hid = numpy.array([400,400,k])
    fc = nn.net_constructor(hid, "relu")
    mlp = mx.sym.SoftmaxOutput(data = fc, name="softmax") #Definiion of the output function

    # Model fit
    model = nn.net_fit(mlp, 10, 0.1, train_iter, val_iter, batch_size, 200, mx.gpu())

    # Predictions and file writing
    pred = nn.predict_neural(test_iter, model)
    scl = numpy.array( range(1,pred.shape[0]+1))
    numpy.savetxt("testOuput.csv", numpy.stack((scl, pred), axis = 1), fmt = '%d', delimiter=",", header = "ImageId,Label", comments='')

if __name__=="__main__":
    main()
