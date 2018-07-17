import numpy as np

# finds the minimum and maximum value of a list of lists (or np arrays)
def findMinMax(listOfLists):
    
    fullList = fullFlatten(listOfLists)
    
    return np.amax(fullList), np.amin(fullList)

# flattens a list of lists (or np arrays) into one long list
def fullFlatten(listOfLists):
    
    flatList = []
    
    for subList in listOfLists:
        flatSubList = np.array(subList).flatten()
        for item in  flatSubList:
            flatList.append(item)
    
    return flatList

# finds the length of each element of the sublist in a list of lists
def subListLengths(listOfLists):
    
    return [len(subList) for subList in listOfLists]