import numpy as np
import matplotlib.pyplot as plt
import time

weights = 10
bias = 1000
x=[10,30,45,25,78,34,64,27,76,34,54,76,23,54,12,35] #15
y=[0,40000,45000,30000,80000,42000,70000,33000,55000,78000,24000,60000,26000,66000,0,43000]
learningRate = 0.00002

def CalculateOutput(input,weights, bias):
    output = (input*weights)+bias
    return output

def CalculateError(expected,predicted):
    return (float(expected)-float(predicted))**2

def CalculateAverageError(x,y):
    averageError = 0.0
    for i in range(0,len(x)):
        predictedOutput = CalculateOutput(x[i],weights,bias)
        error=CalculateError(y[i],predictedOutput)
        averageError+=error
    return averageError/len(x)

def CalculateWeightGradient(x,predicted,expected):
    return -2 * x * (expected - predicted)

def CalculateBiasGradient(predicted,expected):
    return (-2 * (expected - predicted))

def UpdateWeight(gradient):
    global weights
    weights = weights-(learningRate*gradient)

def UpdateBias(gradient):
    global bias
    bias = bias-(learningRate*gradient)

def FeedForward(x,y,epoch):
    for i in range (0,epoch):
        print(i+1,"/",epoch)
        # print("bias before ",bias)
        # print("weight before ",weights)
        avgWeightGradient = 0.0
        avgBiasGradient = 0.0
        for i in range(0,len(x)):
            predictedOutput = CalculateOutput(x[i],weights,bias)
            weightGradient = CalculateWeightGradient(x[i],predictedOutput,y[i])
            biasGradient = CalculateBiasGradient(predictedOutput,y[i])
            avgWeightGradient+=weightGradient
            avgBiasGradient +=biasGradient
        UpdateWeight(avgWeightGradient)
        UpdateBias(avgBiasGradient)
        # print("weight after ",weights)
        # print("bias after ",bias)
        DrawLine(x,y)
        DrawScatter(x,y,"age","salary")
        plt.text(50,100,str(bias),fontsize=12,color='red')
        plt.draw()
        plt.pause(0.01)
        plt.clf()
        # print("weight gradient is ",avgWeightGradient)
        # print("bias gradient is ", avgBiasGradient)

def DrawScatter(x,y,xLabel,yLabel):
    plt.scatter(x,y)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

def DrawLine(x,y):
    # print("weight is " ,weights)
    # print("bias is ", bias)
    lineXMin = min(x)
    lineXMax = max(x)
    xLine =[]
    yLine =[]
    xLine.append(lineXMin)
    yLine.append((weights*lineXMin)+bias)
    xLine.append(lineXMax)
    yLine.append((weights*lineXMax)+bias)
    plt.plot(xLine,yLine)

def predic 