import os
import glob
import pandas as pd
import numpy as np
import csv

class ReadVideoFloders:
    def __init__(self, data_path, output_path, splitratio=0.1):
        self.videos = []
        self.labels = []
        self.data_path = data_path
        self.output_path = output_path
        self.class_to_index = {}
        classFloders = os.listdir(self.data_path)
        self.TotalNumber = len(classFloders) 
        for (i, name) in enumerate(classFloders):
            if self.class_to_index.get(name) is None:
                self.class_to_index[name] = i
            
            self._read_each_class(name)

        # random shuffle
        index = np.arange(self.TotalNumber)
        np.random.shuffle(index)
        self.videos = [self.videos[i] for i in index]
        self.labels = [self.labels[i] for i in index]

        # split dataset into trainset and testset
        self.SplitNumber = int(splitratio*self.TotalNumber)


    def _write_dataset_csv(self, is_train):
        if is_train:
            if os.path.exists(self.output_path):
                with open(os.path.join(self.output_path, 'train.csv'),
                            'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)

                    for i in range(self.SplitNumber):
                        writer.writerow([self.videos[i], 
                                        self.labels[i]])
            else:
                print("The directory is not exist!")
        else:
            if os.path.exists(self.output_path):
                with open(os.path.join(self.output_path, 'test.csv'),
                            'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)

                    for i in range(self.SplitNumber, self.TotalNumber):
                        writer.writerow([self.videos[i], 
                                        self.labels[i]])
            else:
                print("The directory is not exist!")

    def _read_each_class(self, className):
         class_root_path = self.data_path + className
         ClipsFolders = os.listdir(class_root_path) # basketball
         for clipfolder in ClipsFolders:
            video_shootings = glob.glob(os.path.join(class_root_path,
                                        clipfolder, '*.mpg'))
            if video_shootings is not None:
                for vs in video_shootings:
                    self.videos.append(vs)
                    self.labels.append(int(self.class_to_index[className]))

