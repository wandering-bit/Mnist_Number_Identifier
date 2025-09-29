import numpy as np
import cv2
import math
import os
from datasetReader import MakeBatches

batches = MakeBatches()
global_batch=batches.Read()

minRange = -1.0
maxRange = 1.0

inputs = None
hiddenLayer1 = None
hiddenLayer2 = None
hiddenLayer1weight = None
hiddenLayer2weight = None
hiddenLyaer1bias = None
hiddenLyaer2bias = None
outpuLayerWeight = None
outputLayerBias = None
outputLayer = None
learningRate = 0.001
OutputGradientWeight = None
OutputGradientBias = None

Hidden1GradientWeight = None
Hidden1GradientBias = None

Hidden2GradientWeight = None
Hidden2GradientBias = None

np.set_printoptions(suppress=True, precision=6)

def feed_forward(data_inputs, epoch):
    global hiddenLayer1weight,hiddenLyaer1bias,hiddenLayer2weight,hiddenLyaer2bias,outpuLayerWeight,outputLayerBias
    global Hidden1GradientWeight,Hidden1GradientBias,Hidden2GradientWeight,Hidden2GradientBias,OutputGradientWeight,OutputGradientBias
    for epo in range(0,epoch):
        batchcount =0
        print("running epoch ",epo," of ",epoch )
        for batches in data_inputs:
            print("remaining batches: ",len(data_inputs)-batchcount)
            for input in batches:
                imagePath,label = input[0],input[1]
                # print("image path ",imagePath)
                image = cv2.imread(imagePath)
                grayImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                setInput(grayImage)
                setHidden1()
                setHidden2()
                setOutput()
                # print("output is ",outputLayer)
                expected = formExpected(int(label))
                BackPropagate(expected,outputLayer)
            # print("1 batch processed")
            batchcount+=1
            hiddenLayer1weight=hiddenLayer1weight-Hidden1GradientWeight
            hiddenLyaer1bias=hiddenLyaer1bias-Hidden1GradientBias

            hiddenLayer2weight=hiddenLayer2weight-Hidden2GradientWeight
            hiddenLyaer2bias=hiddenLyaer2bias-Hidden2GradientBias

            outpuLayerWeight=outpuLayerWeight-OutputGradientWeight
            outputLayerBias=outputLayerBias-OutputGradientBias
            Hidden1GradientWeight= np.full((900,784),0)
            Hidden1GradientBias=   np.full((900,1),0)
            Hidden2GradientWeight= np.full((900,900),0)
            Hidden2GradientBias=   np.full((900,1),0)
            OutputGradientWeight=  np.full((10,900),0)
            OutputGradientBias=    np.full((10,1),0)

def test(imagePath):
    image = cv2.imread(imagePath)
    grayImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    setInput(grayImage)
    setHidden1()
    setHidden2()
    setOutput()
    print("test output is ", outputLayer)

def BackPropagate(expected,outputLayer):
    global OutputGradientWeight,OutputGradientBias,Hidden2GradientWeight,Hidden2GradientBias,inputs,Hidden1GradientWeight,Hidden1GradientBias
    ##################################################################################
    OuputError = CalculateOutputError(expected,outputLayer)
    # print("output error",OuputError)
    weightGradient = CalculateOutputWeightGradient(hiddenLayer2,OuputError)
    biasGradient = CalculateOuputBiasGradient(OuputError)
    # print("weight gradient size is ", OutputGradientWeight.shape)
    OutputGradientWeight = OutputGradientWeight+weightGradient
    OutputGradientBias = OutputGradientBias+biasGradient
    ##################################################################################
    reshapedWeight = outpuLayerWeight.reshape(900,10)
    error = np.dot(reshapedWeight,OuputError)
    Hidden2Error = CalculateHidden2Error(error)
    # print("hidden2 error shape is ",Hidden2Error.shape)
    weightGradient = CalculateHidden2WeightGradient(hiddenLayer1,Hidden2Error)
    biasGradient = CalculateHidden2BiasGradient(Hidden2Error)
    Hidden2GradientWeight = Hidden2GradientWeight+weightGradient
    Hidden2GradientBias = Hidden2GradientBias+biasGradient
    # print("hidden2 weight gradient shape is ",weightGradient.shape)
    ##################################################################################
    error = np.dot(hiddenLayer2weight,Hidden2Error)
    # print("hidden 1 shape ",error.shape)
    Hidden1Error = CalculateHidden1Error(error)
    weightGradient = CalculateHidden1WeightGradient(inputs,Hidden1Error)
    biasGradient = CalculateHidden1BiasGradient(Hidden1Error)
    Hidden1GradientWeight = Hidden1GradientWeight+weightGradient
    Hidden1GradientBias = Hidden1GradientBias+biasGradient

def CalculateHidden2Error(propagatedError):
    error = hiddenLayer2*(1-hiddenLayer2)
    return error*propagatedError

def CalculateHidden1Error(propagatedError):
    error = hiddenLayer1*(1-hiddenLayer1)
    return error*propagatedError

def UpdateWeight(gradient):
    global weights
    weights = weights-(learningRate*gradient)

def UpdateBias(gradient):
    global bias
    bias = bias-(learningRate*gradient)

def formExpected(trueValue):
    res = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    res[trueValue]=1.0
    res = np.array(res)
    res=res.reshape(10,1)
    return res

def CalculateOutputWeightGradient(input_array,errorArray):
    # print("input is ",input_array)
    # print("error ",errorArray)
    reshapedInput = np.reshape(input_array,(1,900))
    return learningRate * errorArray * reshapedInput

def CalculateOuputBiasGradient(errorArray):
    return learningRate*errorArray

# def CalculateHidden1Gradient(input_array,errorArray):
    
def CalculateHidden2WeightGradient(input_array,errorArray):
    reshapedInput = np.reshape(input_array,(1,900))
    return learningRate*errorArray*reshapedInput

def CalculateHidden2BiasGradient(errorArray):
    return learningRate*errorArray

def CalculateHidden1WeightGradient(input_array,errorArray):
    reshapedInput = np.reshape(input_array,(1,784))
    return learningRate*errorArray*reshapedInput

def CalculateHidden1BiasGradient(errorArray):
    return learningRate*errorArray

def CalculateBiasGradient(errorArray):
    return (-2 * errorArray)

def CalculateError(expected,predicted):
    return expected-predicted

def CalculateOutputError(target,output):
    error =  output*(1-output)*(target-output)
    return error

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def setInput(image):
    global inputs
    k=0
    for i in image:
        for j in i:
            normalizedInput = np.interp(j,[0,255],[0,1])
            inputs[k][0]=normalizedInput
            k+=1

def setHidden1():
    global hiddenLayer1weight,hiddenLyaer1bias,hiddenLayer1
    hidden_layer1 = np.dot(hiddenLayer1weight,inputs)+hiddenLyaer1bias
    for i in range(0,len(hidden_layer1)):
        hiddenLayer1[i][0]=sigmoid(hidden_layer1[i])

def setHidden2():
    global hiddenLayer2weight,hiddenLyaer2bias,hiddenLayer2
    hidden_layer2 = np.dot(hiddenLayer2weight,hiddenLayer1)+hiddenLyaer2bias
    for i in range(0,len(hidden_layer2)):
        hiddenLayer2[i][0]=sigmoid(hidden_layer2[i])

def setOutput():
    global outpuLayerWeight,outputLayerBias,outputLayer
    # print("output layer", outputLayer)
    output_layer = np.dot(outpuLayerWeight,hiddenLayer2)+outputLayerBias
    for i in range(0,len(output_layer)):
        outputLayer[i][0]=sigmoid(output_layer[i])

def initializeInput():
    global inputs
    inputs = np.full((784,1),0.0)

def initializeHidden1():
    global hiddenLayer1weight,hiddenLyaer1bias,hiddenLayer1,Hidden1GradientBias,Hidden1GradientWeight
    hiddenLayer1weight = np.random.uniform(minRange,maxRange,size=(900,784))
    hiddenLyaer1bias = np.random.uniform(minRange,maxRange,size=(900,1))
    hiddenLayer1 = np.full((900,1),0.0)
    Hidden1GradientBias = np.full((900,1),0)
    Hidden1GradientWeight = np.full((900,784),0)

def initializeHidden2():
    global hiddenLayer2weight,hiddenLyaer2bias,hiddenLayer2,Hidden2GradientBias,Hidden2GradientWeight
    hiddenLayer2weight = np.random.uniform(minRange,maxRange,size=(900,900))
    hiddenLyaer2bias = np.random.uniform(minRange,maxRange,size=(900,1))
    hiddenLayer2 = np.full((900,1),0.0)
    Hidden2GradientBias = np.full((900,1),0)
    Hidden2GradientWeight = np.full((900,900),0)

def initializeOuputLayer():
    global outpuLayerWeight,outputLayerBias,outputLayer,OutputGradientBias,OutputGradientWeight
    outpuLayerWeight = np.random.uniform(minRange,maxRange,size=(10,900))
    outputLayerBias = np.random.uniform(minRange,maxRange,size=(10,1))
    outputLayer = np.full((10,1),0.0)
    OutputGradientBias = np.full((10,1),0)
    OutputGradientWeight = np.full((10,900),0)

def main():
    initializeInput()
    initializeHidden1()
    initializeHidden2()
    initializeOuputLayer()
    feed_forward(global_batch,1)
    test(r"D:\Personal\ai\datasets\archive\testSet\testSet\img_15.jpg")
main()