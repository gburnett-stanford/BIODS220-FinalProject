import os 
import argparse 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten, Dense, Input, Dropout, AveragePooling2D
from tensorflow.keras.models import Sequential, Model, save_model
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2 
import tensorflow.keras.backend as K

def main(data_dir, out_dir, batch_size, learning_rate, dropout, unfreeze): 

    # Default VGG16 input size 
    img_width, img_height = 224, 224 

    # Load the dataset
    train_generator = ImageDataGenerator(
        dtype='float32', 
        validation_split=0.2, 
        preprocessing_function=preprocess_input) 

    # Load the dataset as a tf.data.Dataset 
    train_ds = train_generator.flow_from_directory(
        os.path.join(data_dir, 'train'), 
        seed=123,
        subset="training",
        class_mode="categorical",
        target_size=(img_width, img_height), 
        batch_size=batch_size
    )

    val_ds = train_generator.flow_from_directory(
        os.path.join(data_dir, 'train'), 
        seed=123,
        subset="validation",
        class_mode="categorical",
        target_size=(img_width, img_height), 
        batch_size=batch_size
    )

    # Load ResNet50 CNN
    pretrained_model = ResNet50(
        include_top=False, 
        weights="imagenet", 
        input_tensor=Input(shape=(224, 224, 3)))

    # Freeze layers 
    for layer in pretrained_model.layers: 
        # layer.trainable = False
        if unfreeze=="conv5_block3_3":
            if (layer.name=="conv5_block3_add") or (layer.name=="conv5_block3_out"): 
                layer.trainable = True
        if unfreeze in layer.name: 
            print("leaving layer", layer.name, "as trainable")
            layer.trainable = True
        else: 
            layer.trainable = False

    headModel = pretrained_model.output
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dropout(dropout)(headModel)
    headModel = Dense(5, activation="softmax")(headModel)

    model_final = Model(inputs=pretrained_model.input, outputs=headModel)

    print(model_final.summary())
    # return

    # Compile the model 
    opt = Adam(learning_rate=learning_rate) # default = 0.01 
    model_final.compile(
        loss="binary_crossentropy", 
        optimizer = opt, 
        metrics=["accuracy"])

    # Train the model 
    early_stop = EarlyStopping(monitor='val_loss', patience=3)
    history = model_final.fit(
        train_ds, 
        epochs=30, 
        validation_data=val_ds, 
        callbacks=[early_stop]
    )

    # Plot the training results 
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(os.path.join(out_dir, "training_accuracy.png"))

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(os.path.join(out_dir, "training_loss.png"))

    # Load test dataset
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, 'test'), 
        image_size=(224,224), 
        label_mode="categorical", 
        shuffle=False)

    # Predict  
    preds = model_final.predict(test_ds)
    preds = np.argmax(preds, axis=1)
    
    # Calculate accuracy
    labels = np.loadtxt(os.path.join(data_dir, 'test', 'labels.csv'), delimiter=',')
    
    # Compare the predictions 
    match_preds = labels == preds
    acc = sum(match_preds) / len(match_preds)
    print("Test Accuracy:", acc)

    # Save predictions  
    np.savetxt(os.path.join(out_dir, 'predictions.csv'), preds, delimiter=",")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_dir', type=str, help='The directory with the training data', required=True)
    parser.add_argument('--out_dir', type=str, help='The directory to store training files', required=True)
    parser.add_argument('--unfreeze', type=str, help='Which layers to unfreeze', required=True)
    parser.add_argument('--batch_size', type=int, help='The batch size', required=True)
    parser.add_argument('--learning_rate', type=float, help='The learning rate', required=True)
    parser.add_argument('--dropout', type=float, help='The dropout', required=True)

    args = parser.parse_args()

    main(args.data_dir, args.out_dir, args.batch_size, args.learning_rate, args.dropout, args.unfreeze) 