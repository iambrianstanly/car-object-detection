
import tensorflow as tf

def train_model(model, train_ds, valid_ds, config):
    model.compile(loss=[config["loss"], config["bbox_loss"]],
              optimizer=config["optimizer"])
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        config["checkpoint_path"],
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        model="min"
    )
    
    model.fit(train_ds, validation_data=valid_ds, epochs=config["epochs"],callbacks=[checkpoint])


