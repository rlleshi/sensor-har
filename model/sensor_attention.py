import tensorflow as tf


class SensorAttention(tf.keras.layers.Layer):
    """ Captures the varing levels of contribution from sensors at different modalities
        for classification. E.g. to recognize 'ironing', the sensors at the subject's ankle
        do not provide much insight."""

    def __init__(self, n_filters, kernel_size, dilation_rate):
        super(SensorAttention, self).__init__()
        self.conv_f = tf.keras.layers.Conv2D(n_filters, kernel_size=kernel_size,
                                             dilation_rate=dilation_rate, padding='same', activation='relu')
        self.conv_1 = tf.keras.layers.Conv2D(1, kernel_size=1, padding='same')
        self.ln = tf.keras.layers.LayerNormalization()

    def call(self, x):
        # reshape input to produce single channel image
        x = self.ln(x)
        x1 = tf.expand_dims(x, axis=3)
        x1 = self.conv_f(x1)
        # convert back to single-channel
        x1 = self.conv_1(x1)
        # use sensor-wise softmax to provide the attention score for individual sensors
        x1 = tf.keras.activations.softmax(x1, axis=2)
        x1 = tf.keras.layers.Reshape(x.shape[-2:])(x1)
        return tf.math.multiply(x, x1), x1
