from pathlib import Path
import glob
from PIL import Image
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf


preprocess_layer = tf.keras.Sequential([
    tf.keras.layers.Resizing(height=299, width=299),
    tf.keras.layers.Rescaling(scale=1/255)
])



def pipeline(path, label, bbox):
    raw = tf.io.read_file(path)
    img = tf.io.decode_jpeg(raw, channels=3)
    image_tf_preprocessed = preprocess_layer(img)
    bbox_preprocessed = bbox / 299
    return image_tf_preprocessed, (label, bbox_preprocessed)



def get_dataloaders(config):
    
    train_path = Path(config["path"]) / "training_images"

    train_path_pattern = train_path / "*.jpg"

    image_files = sorted(glob.glob(str(train_path_pattern)))

    train_metadata = pd.read_csv(config["path"] + "/train_solution_bounding_boxes (1).csv")

    bounding_box_paths = train_metadata['image']


    full_paths = []
    labels = []
    bboxes = []

    for path in image_files:

        img = Image.open(path)

        full_paths.append(path)

        filename = os.path.basename(path)
        # print("filename:\t", filename)
        flag = 1 if filename in bounding_box_paths.to_list() else 0
        # print(flag)
        if flag:
            # print(f"{filename} contains bounding box")
            labels.append(1.)
            bbox = train_metadata[train_metadata['image'] == filename].drop('image', axis=1)
            bbox = bbox.values.tolist()[0]
            bboxes.append(bbox)

        else:
            labels.append(0.)
            bboxes.append([0., 0. ,0. , 0.])


    Xtrain_paths, Xvalid_paths, ytrain_labels, yvalid_labels, ytrain_bbox, yvalid_bbox = train_test_split(full_paths, 
                                                                                                          labels, 
                                                                                                          bboxes, 
                                                                                                          test_size=config["valid_size"], 
                                                                                                          random_state=config["random_state"])

    train_dataset = tf.data.Dataset.from_tensor_slices((Xtrain_paths, ytrain_labels, ytrain_bbox))
    train_dataset = train_dataset.map(pipeline)
    train_dataset = train_dataset.batch(config["batch_size"])

    valid_dataset = tf.data.Dataset.from_tensor_slices((Xvalid_paths, yvalid_labels, yvalid_bbox))
    valid_dataset = valid_dataset.map(pipeline)
    valid_dataset = valid_dataset.batch(config["batch_size"])


    return train_dataset, valid_dataset
