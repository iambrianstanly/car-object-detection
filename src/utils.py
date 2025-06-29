import random
import numpy as np
import tensorflow as tf


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)




# def preprocess_bounding_box(img, bbox):
#     width, height = img.size
#     width_scale = 299/ width
#     height_scale = 299 / height

#     xmin, ymin, xmax, ymax = bbox

#     xmin_pro = xmin * width_scale
#     y_min_pro = ymin * height_scale
#     xmax_pro = xmax * width_scale
#     ymax_pro = ymax * height_scale

#     return [xmin_pro, y_min_pro, xmax_pro, ymax_pro]

