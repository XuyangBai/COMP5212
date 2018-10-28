import tensorflow as tf

NUM_LABELS = 47
CNN_MODEL_PATH = "./CNN/cnn_model"
AE_MODEL_PATH = "./AE/ae_model"
CNN_PRETRAINED_MODEL = "./CNN_pretrained/cnn_model"
RATIO = 1
PRE_TRAINED = True

def convert_image_data_to_float(image_raw):
    img_float = tf.expand_dims(tf.cast(image_raw, tf.float32) / 255, axis=-1)
    return img_float
