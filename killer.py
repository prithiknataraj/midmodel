import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Step 1: Load and Preprocess the Dataset
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    'path/to/dataset',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    'path/to/dataset',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'path/to/test_dataset',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Step 2: Visualize Samples from the Dataset
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

# Step 3: Load and Modify Pre-Trained Models (VGG16 and ResNet50)
# VGG16 model
vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg16_model = Sequential([
    vgg16_base,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# ResNet50 model
resnet50_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
resnet50_model = Sequential([
    resnet50_base,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Freeze base layers to fine-tune only top layers
for layer in vgg16_base.layers:
    layer.trainable = False
for layer in resnet50_base.layers:
    layer.trainable = False

# Step 4: Compile the Models with Experimented Hyperparameters
learning_rate = 0.0001

vgg16_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
resnet50_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

# Step 5: Train the Models
# VGG16 training
vgg16_history = vgg16_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    batch_size=32
)

# ResNet50 training
resnet50_history = resnet50_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    batch_size=32
)

# Step 6: Fine-Tune the Pre-Trained Models
for layer in vgg16_base.layers[-4:]:  # Fine-tune last 4 layers
    layer.trainable = True
vgg16_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
vgg16_finetune_history = vgg16_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=5,
    batch_size=32
)

for layer in resnet50_base.layers[-4:]:  # Fine-tune last 4 layers
    layer.trainable = True
resnet50_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
resnet50_finetune_history = resnet50_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=5,
    batch_size=32
)

# Step 7: Evaluate the Models on the Test Set
# VGG16 Evaluation
vgg16_test_loss, vgg16_test_accuracy = vgg16_model.evaluate(test_generator)
print(f"VGG16 Test Loss: {vgg16_test_loss}, Test Accuracy: {vgg16_test_accuracy}")
vgg16_y_true = test_generator.classes
vgg16_y_pred = (vgg16_model.predict(test_generator) > 0.5).astype("int32").flatten()
vgg16_cm = confusion_matrix(vgg16_y_true, vgg16_y_pred)

# ResNet50 Evaluation
resnet50_test_loss, resnet50_test_accuracy = resnet50_model.evaluate(test_generator)
print(f"ResNet50 Test Loss: {resnet50_test_loss}, Test Accuracy: {resnet50_test_accuracy}")
resnet50_y_true = test_generator.classes
resnet50_y_pred = (resnet50_model.predict(test_generator) > 0.5).astype("int32").flatten()
resnet50_cm = confusion_matrix(resnet50_y_true, resnet50_y_pred)

# Plot confusion matrices
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Horse", "Human"], yticklabels=["Horse", "Human"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()

plot_confusion_matrix(vgg16_cm, "VGG16")
plot_confusion_matrix(resnet50_cm, "ResNet50")
