import os 
import argparse 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten, Dense, Input, Dropout, AveragePooling2D, Concatenate
from tensorflow.keras.models import Sequential, Model, save_model
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2 
import tensorflow.keras.backend as K

def generate_multiple_inputs(generator, hand_dir, eye_dir, subset, batch_size):   

    hand_train_ds = generator.flow_from_directory(
        hand_dir, 
        seed=123,
        subset=subset,
        class_mode="categorical",
        target_size=(224, 224), # maintain aspect ratio
        batch_size=batch_size
    )

    eye_train_ds = generator.flow_from_directory(
        eye_dir, 
        seed=123,
        subset=subset,
        class_mode="categorical",
        target_size=(224, 224), # maintain aspect ratio
        batch_size=batch_size
    )

    while True:
        X1i = hand_train_ds.next()
        X2i = eye_train_ds.next()
        yield [X1i[0], X2i[0]], X2i[1]  #Yield both images and their mutual label

def generate_multiple_inputs_test(generator, hand_dir, eye_dir, batch_size):   

    hand_train_ds = generator.flow_from_directory(
        hand_dir, 
        seed=123,
        class_mode="categorical",
        target_size=(224, 224), # maintain aspect ratio
        batch_size=batch_size
    )

    eye_train_ds = generator.flow_from_directory(
        eye_dir, 
        seed=123,
        class_mode="categorical",
        target_size=(224, 224), # maintain aspect ratio
        batch_size=batch_size
    )

    while True:
        X1i = hand_train_ds.next()
        X2i = eye_train_ds.next()
        yield [X1i[0], X2i[0]], X2i[1]  #Yield both images and their mutual label

def main(hand_dir, eye_dir, out_dir, batch_size, learning_rate): 

    # Default VGG16 input size 
    img_width, img_height = 224, 224 

    # Load the datasets
    train_generator = ImageDataGenerator(
        dtype='float32', 
        validation_split=0.2, 
        preprocessing_function=preprocess_input) 

    train_ds = generate_multiple_inputs(
        train_generator, 
        os.path.join(hand_dir, 'train'), 
        os.path.join(eye_dir, 'train'), 
        "training", 
        batch_size) 

    val_ds = generate_multiple_inputs(
        train_generator, 
        os.path.join(hand_dir, 'train'), 
        os.path.join(eye_dir, 'train'), 
        "validation", 
        batch_size)

    # Load ResNet50 CNN
    hand_pretrained_model = ResNet50(
        include_top=False, 
        weights="imagenet", 
        input_tensor=Input(shape=(224, 224, 3)))

    # Freeze layers 
    for layer in hand_pretrained_model.layers: 
        if "conv5_block3" in layer.name: 
            print("leaving layer", layer.name, "as trainable")
            layer.trainable = True
        else: 
            layer.trainable = False
        # make unique layer names 
        layer._name = layer._name + "_hand"

    # Load ResNet50 CNN
    eye_pretrained_model = ResNet50(
        include_top=False, 
        weights="imagenet", 
        input_tensor=Input(shape=(224, 224, 3)))

    # Freeze layers 
    for layer in eye_pretrained_model.layers: 
        if "conv5_block3" in layer.name: 
            print("leaving layer", layer.name, "as trainable")
            layer.trainable = True
        else: 
            layer.trainable = False

    headModel = Concatenate()([hand_pretrained_model.output, eye_pretrained_model.output])

    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(5, activation="softmax")(headModel)

    model_final = Model(inputs=[hand_pretrained_model.input, eye_pretrained_model.input], outputs=headModel)

    # print(model_final.summary())
    # return

    # Compile the model 
    opt = Adam(learning_rate=learning_rate) # default = 0.01 
    model_final.compile(
        loss="binary_crossentropy", 
        optimizer = opt, 
        metrics=["accuracy"])

    # Train the model 
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    num_train_imgs = 3573 * 0.8
    num_val_imgs = 3573 * 0.2

    history = model_final.fit(
        train_ds, 
        steps_per_epoch=num_train_imgs/batch_size,
        epochs=30, 
        validation_data=val_ds, 
        validation_steps=num_val_imgs/batch_size,
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
    plt.legend()

    plt.savefig(os.path.join(out_dir, "training_accuracy.png"))

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.savefig(os.path.join(out_dir, "training_loss.png"))

    # Load test dataset
    test_generator = ImageDataGenerator(
        dtype='float32', 
        preprocessing_function=preprocess_input) 

    test_ds = generate_multiple_inputs_test(
        test_generator, 
        os.path.join(hand_dir, 'test'), 
        os.path.join(eye_dir, 'test'), 
        batch_size)

    # Predict  
    preds = model_final.predict(test_ds, steps=893/batch_size)
    preds = np.argmax(preds, axis=1)
    
    # Calculate accuracy
    labels = np.loadtxt(os.path.join(hand_dir, 'test', 'labels.csv'), delimiter=',')
    
    # Compare the predictions 
    match_preds = labels == preds
    acc = sum(match_preds) / len(match_preds)
    print("Test Accuracy:", acc)

    # Save predictions  
    np.savetxt(os.path.join(out_dir, 'predictions.csv'), preds, delimiter=",")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--hand_dir', type=str, help='The directory with the hand training data', required=True)
    parser.add_argument('--eye_dir', type=str, help='The directory with the eye training data', required=True)
    parser.add_argument('--out_dir', type=str, help='The directory to store training files', required=True)
    parser.add_argument('--batch_size', type=int, help='The batch size', required=True)
    parser.add_argument('--learning_rate', type=float, help='The learning rate', required=True)

    args = parser.parse_args()

    main(args.hand_dir, args.eye_dir, args.out_dir, args.batch_size, args.learning_rate) 