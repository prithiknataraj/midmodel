import tensorflow as tf
from tensorflow.keras import layers, models, applications
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, classification_report

# Set the path to the main dataset directory
dataset_dir = 'path/to/dataset'

# Define parameters
batch_size = 32
img_size = (150, 150)
seed = 42

# Load the dataset with an 80-10-10 split (training, validation, testing)
train_val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=img_size,
    batch_size=batch_size
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=img_size,
    batch_size=batch_size
)

# Split validation set into 10% for validation and 10% for testing
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches // 2)
val_dataset = val_dataset.skip(val_batches // 2)

# Visualize samples
def visualize_batch(dataset, title):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title('Horse' if labels[i] == 0 else 'Human')
            plt.axis("off")
    plt.suptitle(title)
    plt.show()

visualize_batch(train_val_dataset, "Training Data")
visualize_batch(val_dataset, "Validation Data")
visualize_batch(test_dataset, "Test Data")

# Define a function to build and fine-tune a model
def build_and_fine_tune_model(base_model, input_shape=(150, 150, 3)):
    base_model.trainable = False  # Freeze the base model layers
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Load two pre-trained models
vgg16_base = applications.VGG16(include_top=False, input_shape=(150, 150, 3))
mobilenet_base = applications.MobileNetV2(include_top=False, input_shape=(150, 150, 3))

# Initialize models
vgg16_model = build_and_fine_tune_model(vgg16_base)
mobilenet_model = build_and_fine_tune_model(mobilenet_base)

# Compile models with different learning rates and optimizers
vgg16_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

mobilenet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

# Train both models
print("Training VGG16 model...")
vgg16_history = vgg16_model.fit(
    train_val_dataset,
    validation_data=val_dataset,
    epochs=5,
    batch_size=batch_size,
    verbose=1
)

print("Training MobileNet model...")
mobilenet_history = mobilenet_model.fit(
    train_val_dataset,
    validation_data=val_dataset,
    epochs=5,
    batch_size=batch_size,
    verbose=1
)

# Plot training & validation accuracy/loss values for both models
def plot_history(history, title):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title(f'{title} Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title(f'{title} Training and Validation Loss')
    plt.legend()

    plt.show()

plot_history(vgg16_history, "VGG16 Model")
plot_history(mobilenet_history, "MobileNetV2 Model")

# Evaluate models on test data
print("\nEvaluating VGG16 model on test data...")
vgg16_test_loss, vgg16_test_acc = vgg16_model.evaluate(test_dataset)
print(f"VGG16 Test Accuracy: {vgg16_test_acc}, Test Loss: {vgg16_test_loss}")

print("\nEvaluating MobileNet model on test data...")
mobilenet_test_loss, mobilenet_test_acc = mobilenet_model.evaluate(test_dataset)
print(f"MobileNet Test Accuracy: {mobilenet_test_acc}, Test Loss: {mobilenet_test_loss}")

# Confusion Matrix and Classification Report on Test Data
def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Function to evaluate and print confusion matrix for a model
def evaluate_model(model, test_dataset, model_name):
    y_true = []
    y_pred = []
    for images, labels in test_dataset:
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend((preds > 0.5).astype(int).flatten())
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, classes=['Horse', 'Human'], title=f'{model_name} Confusion Matrix')
    print(f"\n{model_name} Classification Report:\n", classification_report(y_true, y_pred, target_names=['Horse', 'Human']))

# Evaluate and plot confusion matrix for both models
evaluate_model(vgg16_model, test_dataset, "VGG16 Model")
evaluate_model(mobilenet_model, test_dataset, "MobileNetV2 Model")
