#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 20:27:34 2022

@author: kjayamanna
"""
#%
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from encodeLabels import LabelEncoder
from FeaturePyramids import get_backbone
from loss import RetinaNetLoss
from retinaNet import RetinaNet
from preprocess import preprocess_data
#% Setting up training parameters
model_dir = "retinanet/"
num_classes = 80
batch_size = 2
label_encoder = LabelEncoder()
learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)
#%Initializing and compiling model
def get_model():
    resnet50_backbone = get_backbone()
    loss_fn = RetinaNetLoss(num_classes)
    model = RetinaNet(num_classes, resnet50_backbone)
    
    optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
    model.compile(loss=loss_fn, optimizer=optimizer)
    return model


#%Setting up callbacks
def get_callbacks():
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
            monitor="loss",
            save_best_only=False,
            save_weights_only=True,
            verbose=1,
        )
    ]
#% Setting up a tf.data pipeline
def get_data_pipeline(batch_size):
    (train_dataset, val_dataset), dataset_info = tfds.load(
    "coco/2017", split=["train", "validation"], with_info=True, data_dir="data"
    )
    autotune = tf.data.AUTOTUNE
    train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=autotune)
    train_dataset = train_dataset.shuffle(8 * batch_size)
    train_dataset = train_dataset.padded_batch(
        batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
    )
    train_dataset = train_dataset.map(
        label_encoder.encode_batch, num_parallel_calls=autotune
    )
    train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
    train_dataset = train_dataset.prefetch(autotune)
    
    val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=autotune)
    val_dataset = val_dataset.padded_batch(
        batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True
    )
    val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
    val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
    val_dataset = val_dataset.prefetch(autotune)
    return (train_dataset, val_dataset, dataset_info)

#% Training the model
def train(_epochs):
    epochs = _epochs
    train_dataset, val_dataset, _ = get_data_pipeline(2)
    callbacks_list = get_callbacks()
    model = get_model()
    # Uncomment the following lines, when training on full dataset
    # train_steps_per_epoch = dataset_info.splits["train"].num_examples // batch_size
    # val_steps_per_epoch = \
    #     dataset_info.splits["validation"].num_examples // batch_size
    # train_steps = 4 * 100000
    # epochs = train_steps // train_steps_per_epoch
    # Running 100 training and 50 validation steps,
    # remove `.take` when training on the full dataset
    model.fit(
        train_dataset.take(100),
        validation_data=val_dataset.take(50),
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1,
    )
    return model
