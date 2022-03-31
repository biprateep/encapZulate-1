# @Author: Brett Andrews <andrews>
# @Date:   2019-05-30 13:05:29
# @Last modified by:   andrews
# @Last modified time: 2019-05-30 13:05:54

import time

from tensorflow.keras import callbacks


class TimeHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
