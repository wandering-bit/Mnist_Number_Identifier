import numpy as np
import cv2
from datasetReader import MakeBatches
import time

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

learningRate = 0.01

OutputGradientWeight = None
OutputGradientBias = None

Hidden1GradientWeight = None
Hidden1GradientBias = None

Hidden2GradientWeight = None
Hidden2GradientBias = None

model_save_path = 'model.npz'

accuracy = 0

np.set_printoptions(suppress=True, precision=6)
np.seterr(all="ignore")

def test(data_inputs):
    print(f"\r"+" "*1000,end="",flush=True)
    global hiddenLayer1weight,hiddenLyaer1bias,hiddenLayer2weight,hiddenLyaer2bias,outpuLayerWeight,outputLayerBias
    data = np.load(model_save_path)
    initializeInput()
    initializeHidden1()
    initializeHidden2()
    initializeOuputLayer()
    hiddenLayer1weight = None
    hiddenLyaer1bias   = None
    hiddenLayer2weight = None
    hiddenLyaer2bias   = None
    outpuLayerWeight   = None
    outputLayerBias    = None
    hiddenLayer1weight = data['hidden1weight']
    hiddenLyaer1bias   = data['hidden1bias']
    hiddenLayer2weight = data['hidden2weight']
    hiddenLyaer2bias   = data['hidden2bias']
    outpuLayerWeight   = data['outputweight']
    outputLayerBias    = data['outputbias']
    correct            = 0
    total_count        = len(data_inputs)*10
    counter            = 0
    for batch in data_inputs:
        for input in batch:
            image,label = input[0],input[1]
            image = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
            setInput(image)
            setHidden1()
            setHidden2()
            setOutput()
            predicted = np.argmax(outputLayer)
            if str(predicted) == label:
                correct+=1  
            counter+=1 
        tempacc = correct/total_count
        print(f"\r Tested {counter}/{total_count} | Accuray so far: {tempacc}",end="",flush=True)
    return correct

def feed_forward(data_inputs, epoch):
    global hiddenLayer1weight,hiddenLyaer1bias,hiddenLayer2weight,hiddenLyaer2bias,outpuLayerWeight,outputLayerBias
    global Hidden1GradientWeight,Hidden1GradientBias,Hidden2GradientWeight,Hidden2GradientBias,OutputGradientWeight,OutputGradientBias
    total_batch_count=0
    #this loop goes iterates over the whole data
    for epo in range(0,epoch):
        batchcount =0
        ## this loop iterates over the batches present in the whole data
        for batches in data_inputs:
            start_time = time.time()
            ## this loop iterates over the images in the batch
            for input in batches:
                imagePath,label = input[0],input[1]
                image = cv2.imread(imagePath,cv2.COLOR_BGR2GRAY)
                setInput(image)
                setHidden1()
                setHidden2()
                setOutput()
                expected = formExpected(int(label))
                BackPropagate(expected,outputLayer)
            end_time = time.time()
            time_taken = end_time-start_time
            time_taken_minute = time_taken/60
            batchcount+=1
            total_batch_count+=1
            percentage = batchcount/len(data_inputs)*100
            totalPercentage = total_batch_count/(len(data_inputs)*epoch)*100
            print(f"\r|Epoch Completion: {percentage:.2f}% | Total Completion: {totalPercentage:.2f}% | Epoch: {epo} Of {epoch} | Total Time Remaining: {(time_taken_minute*((len(data_inputs)*epoch)-total_batch_count)):.2f} Minutes",end="",flush=True)
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

            np.savez(model_save_path,
                hidden1weight=hiddenLayer1weight,
                hidden1bias  =hiddenLyaer1bias,
                hidden2weight=hiddenLayer2weight,
                hidden2bias  =hiddenLyaer2bias,
                outputweight =outpuLayerWeight,
                outputbias   =outputLayerBias)

def BackPropagate(expected,outputLayer):
    global OutputGradientWeight,OutputGradientBias,Hidden2GradientWeight,Hidden2GradientBias,inputs,Hidden1GradientWeight,Hidden1GradientBias
    ##################################################################################
    OuputError = CalculateOutputError(expected,outputLayer)
    weightGradient = CalculateOutputWeightGradient(hiddenLayer2,OuputError)
    biasGradient = CalculateOuputBiasGradient(OuputError)
    OutputGradientWeight = OutputGradientWeight+weightGradient
    OutputGradientBias = OutputGradientBias+biasGradient
    ##################################################################################
    reshapedWeight = outpuLayerWeight.reshape(900,10)
    error = np.dot(reshapedWeight,OuputError)
    Hidden2Error = CalculateHidden2Error(error)
    weightGradient = CalculateHidden2WeightGradient(hiddenLayer1,Hidden2Error)
    biasGradient = CalculateHidden2BiasGradient(Hidden2Error)
    Hidden2GradientWeight = Hidden2GradientWeight+weightGradient
    Hidden2GradientBias = Hidden2GradientBias+biasGradient
    ##################################################################################
    error = np.dot(hiddenLayer2weight,Hidden2Error)
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
    reshapedInput = np.reshape(input_array,(1,900))
    return learningRate * errorArray * reshapedInput

def CalculateOuputBiasGradient(errorArray):
    return learningRate*errorArray
    
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

def CalculateOutputError(expected,predicted):
    error = predicted-expected
    return error

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def softMax(logits):
    exp_values = np.exp(logits - np.max(logits))
    probabilities = exp_values / np.sum(exp_values)
    return probabilities

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
    output_layer = np.dot(outpuLayerWeight,hiddenLayer2)+outputLayerBias
    outputLayer=softMax(output_layer)

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
    correct = test(global_batch)
    print("accuracy is ", correct/len(global_batch)*10)

main()