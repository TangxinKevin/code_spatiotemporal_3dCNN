from __future__ import print_function
import sys
import os
import csv
import numpy as np

from PIL import Image
import imageio
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, MaxPool3D, Dropout, Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.utils import multi_gpu_model


abs_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(abs_path, "DataSets", "UCF101")
model_path = os.path.join(abs_path, "Models")

# Define the reader for both training and evaluation action.
class VideoReader(object):
    '''
    A simple VideoReader:
    It iterates through each video and select 16 frames as 
    stacked numpy arrays.
    '''
    def __init__(self, map_file, label_count, is_training, limit_epoch_size=sys.maxsize):
        '''
        Load video file paths and their corresponding labels.
        '''
        self.map_file = map_file
        self.label_count = label_count
        self.width = 112
        self.height = 112
        self.sequence_length = 16
        self.channel_count = 3
        self.is_training = is_training
        self.video_files = []
        self.targets = []
        self.batch_start = 0

        map_file_dir = os.path.dirname(map_file)

        with open(map_file) as csv_file:
            data = csv.reader(csv_file)
            for row in data:
                self.video_files.append(os.path.join(map_file_dir, row[0]))
                target = [0.0] * self.label_count
                target[int(row[1])] = 1.0
                self.targets.append(target)

        self.indices = np.arange(len(self.video_files))
        if self.is_training:
            np.random.shuffle(self.indices)
        self.epoch_size = min(len(self.video_files), limit_epoch_size)

    def size(self):
        return self.epoch_size

    def has_more(self):
        if self.batch_start < self.size():
            return True
        return False

    def reset(self):
        if self.is_training:
            np.random.shuffle(self.indices)
        self.batch_start = 0

    def next_minibatch(self, batch_size):
        '''
        Return a mini batch of sequence frames and their corresponding ground truth
        '''
        batch_end = min(self.batch_start + batch_size, self.size())
        current_batch_size = batch_end - self.batch_start
        if current_batch_size < 0:
            raise Exception('Reach the end of the training data.')

        inputs = np.empty(shape=(current_batch_size, self.channel_count, self.sequence_length,
                                 self.height, self.width), dtype=np.float32)
        targets = np.empty(shape=(current_batch_size, self.label_count), dtype=np.float32)
        for idx in range(self.batch_start, batch_end):
            index = self.indices[idx]
            inputs[idx - self.batch_start, :, :, :, :] = self._select_features(self.video_files[index])
            targets[idx - self.batch_start, :] = self.targets[idx]

        self.batch_start += current_batch_size
        return inputs, targets, current_batch_size

    
    def _select_features(self, video_file):
        '''
        Select a sequence of frames from video_file and return them as a Tensor.
        '''
        video_reader = imageio.get_reader(video_file, 'ffmpeg')
        num_frames = len(video_reader)
        if self.sequence_length > num_frames:
            raise ValueError('Sequence length {} is larger than the total number of frames {} in {}'.formt(
                self.sequence_length, num_frames, video_file))

        # select wich sequence frames to use.
        step = int(num_frames / self.sequence_length)
        frame_range = [step * i for i in range(1, self.sequence_length + 1)]
        video_frames = []
        for frame_index in frame_range:
            video_frames.append(self._read_frame(video_reader.get_data(frame_index))
        
        return np.stack(video_frames, axis=1)   # channel, sequence_length, height, width

    
    def _read_frame(self, data):
        '''
        We resize the image to 128x171 first, then selecting a 112x112 crop.
        '''
        if (self.width >= 171) or (self.height >= 128):
            raise ValueError("Target width need to be less than 171 and target height need to be less than 128.")

        image = Image.fromarray(data)
        image.thumbnail((171, 128), Image.ANTIALIAS)

        center_w = image.size[0] / 2
        center_h = image.size[1] / 2

        image = image.crop((center_w - self.width / 2,
                            center_h - self.height / 2,
                            center_w + self.width / 2,
                            center_h + self.height / 2))
        
        norm_image = np.array(image, dtype=np.float32)
        norm_image -= 127.5
        norm_image /= 127.5

        # (channel, height, width)
        return np.ascontiguousarray(np.transpose(norm_image, (2, 0, 1)))


def Conv3D_model(input_shape, num_classes, temporal_depth, kernel_size=3):
    '''
    Define a convolutional 3D network for action recognition.
    input_shape = (, channel, sequence_length, height, width)
    temporal_depth is a list
    '''
    video_input = Input(shape=input_shape)
    for i in range(5):
        x = Conv3D(64 * (2 ** i), 
                   (temporal_depth[i], kernel_size, kernel_size), 
                    padding="same",
                    activation="relu",
                    data_format="channel_first", 
                    name="Conv_block_" + str(i + 1) + "_conv3d")(video_input)
        if i == 0:
            x = MaxPool3D(pool_size=(1, 2, 2), strides=(1, 1, 1),
                            name="Conv_block_" + str(i + 1) + "_maxpooling3d")(x)
        else:
            x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                            name="Conv_block_" + str(i + 1) + "_maxpooling3d")(x)
    
    x = Flatten(name="flatten_layer")(x)
    for i in range(2):
        x = Dense(2048, 
                  activation="relu",
                  name="Dense_" + str(i + 1))(x)
        x = Dropout(0.4)(x)
    prediction = Dense(num_classes,
                        activation='softmax',
                        name="output")(x)
    
    model = Model(inputs=video_input, outputs=prediction)
    model.summary()
    return model

def lr_schedule(epoch):
    """
    Learning Rate Schedule

    Learning rate is scheduled to be reduced after 10, 20
    :param epoch: The number of epochs
    :return: lr (float32) learning rate
    """
    lr = 0.003

    if (epoch + 1) % 4 == 0:
        lr /= 10
    print('Learning rate: ', lr)
    return lr


    

def conv3D_UCF101(train_reader, test_reader, batch_size, temporal_depth, max_epochs=30):

    # These values must match for both train and test reader
    image_height = train_reader.height
    image_width = train_reader.width
    num_channels = train_reader.channel_count
    sequence_length = train_reader.sequence_length
    num_output_classes = train_reader.label_count

    train_minibatch_size = batch_size

    input_shape = (None, num_channels, sequence_length, image_height, image_width)

    model = Conv3D_model(input_shape, num_output_classes, temporal_depth)
    
    try:
        model = multi_gpu_model(model, cpu_merge=False)
        print("Training using multiple GPUs...")
    except:
        print("Training using single GPU or CPU ...")

    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=lr_schedule(0)),
                  loss=categorical_crossentropy,
                  metrics=[categorical_accuracy])

    for epoch in range(max_epochs):
        train_reader.reset()

        while train_reader.has_more():
            videos, labels, current_minibatch = train_reader.next_minibatch(train_minibatch_size)
            tr_loss, tr_acc = model.train_on_batch(videos, lbales)
            print('Epoch {}: Loss {}, Acc {}'.format(epoch, tr_loss, tr_acc))

    # Test data for trained model
    test_minibatch_size = batch_size

    # process minibatches and evaluate the model
    metric_numer = 0
    metric_denom = 0
    minibatch_index = 0

    test_reader.reset()
    while test_reader.has_more():
        videos, labels, current_minibatch = test_reader.next_minibatch(test_minibatch_size)
        _, tt_acc = model.test_on_batch(videos, labels)
        metric_numer += tt_acc * current_minibatch
        metric_denom += current_minibatch
        minibatch_index += 1

    print("")
    print("Final Results: Minibatch[1-{}]: errs = {:0.2f}% * {}".format(minibatch_index+1, 
            (metric_numer*100.0)/metric_denom, metric_denom))
    print("")

    return metric_numer/metric_denom

if __name__ == '__main__':
    num_output_classes = 11
    train_reader = VideoReader(os.path.join(data_path, 'train_map.csv'),
                                num_output_classes, True)
    test_reader = VideoReader(os.path.join(data_path, 'test_map.csv'), 
                                num_output_classes, False)
    batch_size = 30
    temporal_depth = [3, 3, 5, 5, 7]
    conv3D_UCF101(train_reader, test_reader, temporal_depth)


