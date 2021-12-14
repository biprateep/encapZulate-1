import tensorflow as tf
from keras import backend as K


def mqe(y_true, y_pred):
    error = y_true - y_pred
    return K.pow(error, 4)


def central_mse(num_quantiles, **kwargs):
    def central_mse_metric(y_true, y_pred):
        # y_true, weights = tf.split(y_true, 2, axis=-1)
        if kwargs["logistic"]:
            y_true = (K.exp(y_true) * kwargs["z_max"] + kwargs["z_min"]) / (
                K.exp(y_true) + 1
            )
            y_pred = (K.exp(y_pred) * kwargs["z_max"] + kwargs["z_min"]) / (
                K.exp(y_pred) + 1
            )
        if num_quantiles:
            loc = (num_quantiles // 2) + 1
            y_central = tf.split(y_pred, num_quantiles, -1)[loc]
            error = y_central - y_true
        else:
            error = y_true - y_pred
        return K.square(error)

    return central_mse_metric


def central_bias(num_quantiles, **kwargs):
    def central_bias_metric(y_true, y_pred):
        # y_true, weights = tf.split(y_true, 2, axis=-1)
        if kwargs["logistic"]:
            y_true = (K.exp(y_true) * kwargs["z_max"] + kwargs["z_min"]) / (
                K.exp(y_true) + 1
            )
            y_pred = (K.exp(y_pred) * kwargs["z_max"] + kwargs["z_min"]) / (
                K.exp(y_pred) + 1
            )
        if num_quantiles:
            loc = (num_quantiles // 2) + 1
            y_central = tf.split(y_pred, num_quantiles, -1)[loc]
            error = (y_true - y_central) / (1 + y_true)
        else:
            error = (y_true - y_pred) / (1 + y_true)
        return error

    return central_bias_metric


def quantile_loss(num_quantiles, **kwargs):
    def quantile_loss_metric(y_true, y_pred):
        q_all = tf.linspace(0.0, 1.0, num_quantiles + 2)
        q = q_all[1:-1]
        # y_true, weights = tf.split(y_true, 2, axis=-1)
        error = y_true - y_pred

        return K.mean(K.mean(K.maximum(q * error, (q - 1) * error), axis=-1))

    return quantile_loss_metric


def crps_integer_label(y_true, y_pred):
    """Continuous Ranked Probability Score.
       Tensor implementation of CRPS where the labels are integers.
    Args:
        y_true (tensor): [None, n_classes]
        y_pred (tensor): [None, num_capsule]

    Returns:
        The mean CRPS value for the batch scaled in terms of integer labels.
    """
    # get the heaviside function (CDF of true value) using the one hot encodings
    heaviside = K.cumsum(y_true, axis=-1)

    # get integer encodings from one hot encodings
    y_true = K.argmax(y_true, axis=-1)
    y_true = K.expand_dims(K.cast(y_true, "float32"), axis=-1)

    # predicted CDF
    y_cdf_pred = K.cumsum(y_pred, axis=-1)

    # return the mean CRPS value
    return K.mean(K.sum(K.square(y_cdf_pred - heaviside), axis=-1))


def crps_z_label(dz):
    """Continuous Ranked Probability Score.
       Tensor implementation of CRPS where the labels are converted to redhshifts.
    Args:
        z_min (float): Minimum of the redshift range
        dz (float): Width of the redshift bins

        The following are required by the function which this function calls:
            y_true (tensor): [None, n_classes]
            y_pred (tensor): [None, num_capsule]

    Returns:
        The mean CRPS value for the batch scaled in terms of redshift labels.
    """

    def crps_loss(y_true, y_pred):

        """Continuous Ranked Probability Score.
           Tensor implementation of CRPS where the labels are converted to redhshifts.
        Args:
            y_true (tensor): [None, n_classes]
            y_pred (tensor): [None, num_capsule]

        Returns:
            The mean CRPS value.
        """
        # get the heaviside function (CDF of true value) using the one hot encodings
        heaviside = K.cumsum(y_true, axis=-1)

        # get integer encodings from one hot encodings
        y_true = K.argmax(y_true, axis=-1)
        y_true = K.expand_dims(K.cast(y_true, "float32"), axis=-1)

        # predicted CDF
        y_cdf_pred = K.cumsum(y_pred, axis=-1)

        # return the mean CRPS value
        return K.mean(K.sum(K.square(y_cdf_pred - heaviside), axis=-1) * dz)

    return crps_loss


def mse_prob_max(y_true, y_pred):
    """Mean Squared error given the probability distributions.
       The peak of the probability is used as the prediction for training.
    Args:
        y_true (tensor): [None, n_classes]
        y_pred (tensor): [None, num_capsule]

    Returns:
        Scalar loss value.
    """

    # Convert one hot encodings to integer encodings for MSE
    # argmax returns an int while mean and square expect a float, so typecasting is neccessary
    y_true = K.cast(K.argmax(y_true, axis=-1), "float32")
    y_pred = K.cast(K.argmax(y_pred, axis=-1), "float32")
    # calculate the mean squared error
    return K.mean(K.square(y_pred - y_true), axis=-1)


def mse_prob_weighted(y_true, y_pred):
    """Mean Squared error given the probability distributions.
       The weighted sum of the prediction is used to calculate the MSE.
    Args:
        y_true (tensor): [None, n_classes]
        y_pred (tensor): [None, num_capsule]

    Returns:
        Scalar loss value.
    """
    # Convert one hot encodings to integer encodings
    y_true = K.cast(K.argmax(y_true, axis=-1), "float32")

    num_capsule = K.int_shape(y_pred)[-1]
    labels = K.arange(0, stop=num_capsule, step=1, dtype="float32")
    # convert class labels into weighted sum
    y_pred = K.sum(y_pred * labels, axis=-1)

    return K.mean(K.square(y_pred - y_true), axis=-1)


def margin_loss(y_true, y_pred):
    """Margin loss.

    Margin loss from Eq.(4). When y_true[i, :] contains not just one
    `1`, this loss should work too but that has not been tested.

    Args:
        y_true (tensor): [None, n_classes]
        y_pred (tensor): [None, num_capsule]

    Returns:
        Scalar loss value.
    """
    L = (y_true * K.square(K.maximum(0.0, 0.9 - y_pred))) + (
        0.5 * (1 - y_true) * K.square(K.maximum(0.0, y_pred - 0.1))
    )

    return K.mean(K.sum(L, 1))


###############Needs to be relooked at###############
def kl_div(pred_dist, true_dist, normalize=False):
    # SEE KL div already in keras
    """Kullback-Leibler Divergence for discrete distributions.
    Keras based tensor implementation.

    Args:
        pred_dist (tensor): the input probability distribution
        true_dist (tensor): The reference distribution
        normalize (Boolean): Normalize the given distributions
    Returns:
        The K-L Divergence value between the input and the true distributions
    """
    if normalize == True:
        pred_dist = pred_dist / K.sum(pred_dist)
        true_dist = true_dist / K.sum(true_dist)

    return K.sum(true_dist * K.log(true_dist / pred_dist))


def pit(pdf, return_pdf=True, normalize=True):
    """Caluculate the Probability integral transform for a given discrete pdf
    Keras based tensor implementation

    Args:
        pdf (tensor): Input pdf
        return_pdf (Boolean): whether to return the pdf of the PIT or just the PIT
        normalize (Boolean): whether or not to normalize the input pdf before integrating
    Returns:
        if return_pdf is True:
        if return_pdf is False:
    """
    pass
