import tensorflow as tf


def get_model(config):
    base_model = tf.keras.applications.xception.Xception(weights=config["weights"], include_top=False)
    avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

    # output of class
    cls_out = tf.keras.layers.Dense(config["num_classes"], activation=config["output_activation"], name="cls_output")(avg)

    # output of boundingbox
    bbox_out = tf.keras.layers.Dense(4, name="bbox_output")(avg)

    model = tf.keras.Model(inputs=base_model.input, outputs=[cls_out, bbox_out])

    return model
    
