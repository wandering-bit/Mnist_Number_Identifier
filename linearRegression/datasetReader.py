import cv2
import os
import random


class MakeBatches():
    def Read(self):
        parentDirectory = r"C:\Users\Ranjan\Personal\Mnist_Number_Identifier\datasets\archive\trainingSet1\trainingSet"
        batchSize=10
        folders = os.listdir(parentDirectory)
        fileStructure = {}
        inputSize = 0.0
        print(folders)
        for folder in folders:
            files = os.listdir(os.path.join(parentDirectory,folder))
            filess = []
            for file in files:
                filess.append(os.path.join(parentDirectory,folder,file))
                inputSize+=1
            fileStructure[folder]=filess

        batches = []
        inserted=0
        batchIndex=0
        while True:
            batches.append([])
            for category in fileStructure:
                if len(fileStructure[category])==0:
                    continue
                randIndex = random.randint(0,len(fileStructure[category])-1)
                batches[batchIndex].append((fileStructure[category][randIndex],category))
                fileStructure[category].pop(randIndex)
                inserted+=1
            batchIndex+=1
            if inserted==inputSize:
                break
        return batches

def main():
    batch = MakeBatches()
    batch.Read()
if __name__ == "__main__":
    main()