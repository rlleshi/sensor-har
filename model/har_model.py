import tensorflow as tf

from .attentive_pooling import AttentionWithContext
from .self_attention.encoder import EncoderLayer
from .self_attention.positional_encoding import PositionalEncoding
from .sensor_attention import SensorAttention


def create_model(n_timesteps, n_features, n_outputs, _dff=512, d_model=128, nh=4, dropout_rate=0.2, use_pe=False):
    print('===== Model Params =====')
    print(n_timesteps, n_features, n_outputs, _dff, d_model, nh, dropout_rate, use_pe)

    # input consists of timestamps windows of sensor data
    inputs = tf.keras.layers.Input(shape=(n_timesteps, n_features,))

    # apply sensor modality to get a weighted representation of the sensor values
    # according to their attention score. This learned attention score represents
    # the contribution of each of the sensor modalities in the feature
    # representation used by subsequent layers
    si, _ = SensorAttention(n_filters=128, kernel_size=3, dilation_rate=2)(inputs)

    # convert the weighted sensor values to `d` size vectors over single time-steps
    x = tf.keras.layers.Conv1D(d_model, 1, activation='relu')(si)

    if use_pe:
        x *= tf.math.sqrt(tf.cast(d_model, tf.float32))
        x = PositionalEncoding(n_timesteps, d_model)(x)
        x = tf.keras.layers.Dropout(rate=dropout_rate)(x)

    x = EncoderLayer(d_model=d_model, num_heads=nh, dff=_dff, rate=dropout_rate)(x)
    x = EncoderLayer(d_model=d_model, num_heads=nh, dff=_dff, rate=dropout_rate)(x)
    # x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = AttentionWithContext()(x)
    # x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(n_outputs * 4, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    # x = tf.keras.layers.Dense(128, activation='relu') (x)

    predictions = tf.keras.layers.Dense(n_outputs, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)

    return model
