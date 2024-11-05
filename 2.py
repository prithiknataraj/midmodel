import tensorflow as tf
from tensorflow.keras.applications import VGG16, MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'path/to/dataset',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'path/to/dataset',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)


def visualize_samples(generator):
    images, labels = next(generator)
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i])
        plt.title("Horse" if labels[i] == 0 else "Human")
        plt.axis('off')
    plt.show()

visualize_samples(train_generator)


def build_pretrained_model(base_model, num_neurons=128):
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(num_neurons, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

# Initialize pre-trained models
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
mobilenet_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Freeze base models
vgg_base.trainable = False
mobilenet_base.trainable = False

# Build models
vgg_model = build_pretrained_model(vgg_base)
mobilenet_model = build_pretrained_model(mobilenet_base)


def compile_and_train(model, train_gen, val_gen, learning_rate=0.001, batch_size=32, epochs=10):
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return history


vgg_history = compile_and_train(vgg_model, train_generator, validation_generator, learning_rate=0.001, epochs=10)


mobilenet_history = compile_and_train(mobilenet_model, train_generator, validation_generator, learning_rate=0.001, epochs=10)


def fine_tune_model(model, base_model, learning_rate=0.0001):
    # Unfreeze the base model layers
    base_model.trainable = True

    # Compile the model with a lower learning rate
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

vgg_model = fine_tune_model(vgg_model, vgg_base)
mobilenet_model = fine_tune_model(mobilenet_model, mobilenet_base)

# Re-train with fine-tuning
vgg_fine_history = vgg_model.fit(train_generator, validation_data=validation_generator, epochs=5)
mobilenet_fine_history = mobilenet_model.fit(train_generator, validation_data=validation_generator, epochs=5)



test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'path/to/test_dataset',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Evaluate VGG16
vgg_test_loss, vgg_test_accuracy = vgg_model.evaluate(test_generator)
print(f"VGG16 Test Loss: {vgg_test_loss}, VGG16 Test Accuracy: {vgg_test_accuracy}")

# Evaluate MobileNetV2
mobilenet_test_loss, mobilenet_test_accuracy = mobilenet_model.evaluate(test_generator)
print(f"MobileNetV2 Test Loss: {mobilenet_test_loss}, MobileNetV2 Test Accuracy: {mobilenet_test_accuracy}")


# Generate predictions
def plot_confusion_matrix(model, generator, title="Confusion Matrix"):
    y_true = generator.classes
    y_pred = (model.predict(generator) > 0.5).astype("int32").flatten()
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Horse", "Human"], yticklabels=["Horse", "Human"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# Confusion matrices
plot_confusion_matrix(vgg_model, test_generator, title="VGG16 Confusion Matrix")
plot_confusion_matrix(mobilenet_model, test_generator, title="MobileNetV2 Confusion Matrix")
