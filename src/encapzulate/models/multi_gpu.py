from keras import Model
from keras.utils import multi_gpu_model


class MultiGPUModel(Model):
    """Fixes checkpoint saving for multi-gpu models.

    https://github.com/keras-team/keras/issues/2436#issuecomment-354882296
    """

    def __init__(self, serial_model, gpus):
        parallel_model = multi_gpu_model(serial_model, gpus)
        self.__dict__.update(parallel_model.__dict__)
        self._serial_model = serial_model

    def __getattribute__(self, attrname):
        """Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        """
        # return Model.__getattribute__(self, attrname)
        if ("load" in attrname) or ("save" in attrname):
            return getattr(self._serial_model, attrname)

        return super(MultiGPUModel, self).__getattribute__(attrname)
