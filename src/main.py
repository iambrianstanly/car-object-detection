from fastapi import FastAPI, UploadFile
import tensorflow as tf
import numpy as np



model = tf.keras.models.load_model("models/best_model")


app = FastAPI()

preprocess_layer = tf.keras.Sequential([
    tf.keras.layers.Resizing(height=299, width=299),
    tf.keras.layers.Rescaling(scale=1/255)
])


def pipeline(image_bytes):
    img = tf.io.decode_jpeg(image_bytes, channels=3)
    image_tf_preprocessed = preprocess_layer(img)
    return image_tf_preprocessed


def serialize_to_json(predict):
    label = predict[0][0]
    label = 1 if label > 0.5 else 0
    bbox = predict[1][0].tolist()
    return label, bbox

def denormalize_bbox(bbox):
        xmin_pred, ymin_pred, xmax_pred, ymax_pred = bbox[0], bbox[1], bbox[2], bbox[3]
        xmin_pred, ymin_pred, xmax_pred, ymax_pred = xmin_pred * 299,ymin_pred * 299, xmax_pred*299, ymax_pred*299
        return {"xmin":xmin_pred,
                "ymin":ymin_pred,
                "xmax":xmax_pred,
                "ymax":ymax_pred}



@app.get("/")
def welcome():
    return {"message": "Welcome"}


@app.post("/predict/")
async def create_file(file: UploadFile):
    image_bytes = await file.read()
    img = pipeline(image_bytes)
    img = tf.expand_dims(img, axis=0)
    y_predict = model.predict(img)
    label, bbox = serialize_to_json(y_predict)
    if label:
        bbox_denormalized = denormalize_bbox(bbox)
        return {"Detect":"Car detected", "bounding box":bbox_denormalized}
    else:
        return {"Detect":"No car detected"}

