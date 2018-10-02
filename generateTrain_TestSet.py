import os
import pandas as pd

data_path = '/home/user/Documents/dataset/UCF101/ucfTrainTestlist'
testFiles = ['testlist0{}'.format(i) for i in range(1, 4)]
trainFiles = ['trainlist0{}'.format(i) for i in range(1, 4)]
classIndexFile = 'classInd.txt'

