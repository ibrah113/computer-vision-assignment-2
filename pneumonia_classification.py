from __future__ import print_function

import os
import keras
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Rescaling
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix


batch_size = 12
num_classes = 3
epochs = 8
img_width = 128
img_height = 128
img_channels = 3
fit = True  # make fit false if you do not want to train the network again
train_dir = r'C:\Users\ibrah\Semester 2\Computer Vision\Assignment2\chest_xray\train'
test_dir = r'C:\Users\ibrah\Semester 2\Computer Vision\Assignment2\chest_xray\test'


with tf.device('/cpu:0'):

    # create training, validation and test datasets
    train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        seed=123,
        validation_split=0.2,
        subset='both',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        shuffle=True
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        seed=None,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        shuffle=False
    )

    class_names = train_ds.class_names
    print('Class Names: ', class_names)
    num_classes = len(class_names)

    def count_images(folder):
        counts = {}
        for cls in os.listdir(folder):
            cls_path = os.path.join(folder, cls)
            if os.path.isdir(cls_path):
                counts[cls] = len(os.listdir(cls_path))
        return counts

    train_counts = count_images(train_dir)
    test_counts = count_images(test_dir)

    print("Train distribution:", train_counts)
    print("Test distribution:", test_counts)

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i].numpy()])
            plt.axis("off")
    plt.show()

    # baseline model + class weighting
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(img_height, img_width, img_channels)),
        Rescaling(1.0 / 255),

        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy']
    )

    save_callback = tf.keras.callbacks.ModelCheckpoint(
        "pneumonia.keras",
        save_freq='epoch',
        save_best_only=True
    )

    # get labels from training dataset
    y_train_labels = []
    for images, labels in train_ds:
        y_train_labels.extend(labels.numpy())

    y_train_labels = np.array(y_train_labels)

    # compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_labels),
        y=y_train_labels
    )

    class_weight_dict = dict(enumerate(class_weights))

    # manually boost viral class (index 2)
    class_weight_dict[2] *= 1.3

    print("Class Weights:", class_weight_dict)

    if fit:
        history = model.fit(
            train_ds,
            batch_size=batch_size,
            validation_data=val_ds,
            callbacks=[save_callback],
            epochs=epochs,
            class_weight=class_weight_dict
        )
    else:
        model = tf.keras.models.load_model("pneumonia.keras")

    score = model.evaluate(test_ds, batch_size=batch_size)
    print('Test accuracy:', score[1])

    if fit:
        plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    test_batch = test_ds.take(1)
    plt.figure(figsize=(10, 10))
    for images, labels in test_batch:
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            prediction = model.predict(tf.expand_dims(images[i].numpy(), 0), verbose=0)
            plt.title(
                'Actual:' + class_names[labels[i].numpy()] +
                '\nPredicted:{} {:.2f}%'.format(class_names[np.argmax(prediction)], 100 * np.max(prediction))
            )
            plt.axis("off")
    plt.show()

    # full test-set metrics
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))