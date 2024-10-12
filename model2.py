import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras.applications import ResNet50
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import cv2  # For image resizing

# Load data
train_df = pd.read_csv("sign_mnist_train.csv")
test_df = pd.read_csv("sign_mnist_test.csv")

y_train = train_df['label']
y_test = test_df['label']
del train_df['label']
del test_df['label']

label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.transform(y_test)

x_train = train_df.values
x_test = test_df.values

x_train = x_train / 255
x_test = x_test / 255

# Reshape to grayscale
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convert grayscale images to RGB
x_train_rgb = np.repeat(x_train, 3, axis=-1)
x_test_rgb = np.repeat(x_test, 3, axis=-1)

# Resize images from 28x28 to 32x32
x_train_rgb = np.array([cv2.resize(img, (32, 32)) for img in x_train_rgb])
x_test_rgb = np.array([cv2.resize(img, (32, 32)) for img in x_test_rgb])

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1, 
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False)
datagen.fit(x_train_rgb)

# Load pre-trained ResNet50 model without top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False

# Build the model
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(24, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', 
              metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
model.summary()

# Define custom callback for additional metrics
class MetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            y_pred = self.model.predict(x_test_rgb)
            y_pred_classes = y_pred.argmax(axis=-1)
            y_true_classes = y_test.argmax(axis=-1)

            precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
            recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
            f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

            logs['val_precision'] = precision
            logs['val_recall'] = recall
            logs['val_f1'] = f1

metrics_callback = MetricsCallback()

# Train the model
history = model.fit(datagen.flow(x_train_rgb, y_train, batch_size=128), epochs=100, validation_data=(x_test_rgb, y_test), callbacks=[metrics_callback])

# Compute F1 score
def compute_f1_score(y_true, y_pred):
    y_pred_classes = y_pred.argmax(axis=-1)
    y_true_classes = y_true.argmax(axis=-1)
    return f1_score(y_true_classes, y_pred_classes, average='weighted')

history_dict = history.history

y_pred = model.predict(x_test_rgb)
f1_scores = [compute_f1_score(y_test, y_pred) for _ in range(len(history_dict.get('accuracy', [])))]

# Plot results
sns.set(style="whitegrid")

plt.figure(figsize=(20, 15))

plt.subplot(3, 2, 1)
sns.lineplot(x=range(1, len(history_dict['loss']) + 1), y=history_dict['loss'], label='Training Loss')
sns.lineplot(x=range(1, len(history_dict['val_loss']) + 1), y=history_dict['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.subplot(3, 2, 2)
sns.lineplot(x=range(1, len(history_dict['accuracy']) + 1), y=history_dict['accuracy'], label='Training Accuracy')
sns.lineplot(x=range(1, len(history_dict['val_accuracy']) + 1), y=history_dict['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.subplot(3, 2, 3)
sns.lineplot(x=range(1, len(history_dict.get('precision', [])) + 1), y=history_dict.get('precision', []), label='Training Precision')
sns.lineplot(x=range(1, len(history_dict.get('val_precision', [])) + 1), y=history_dict.get('val_precision', []), label='Validation Precision')
plt.title('Model Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend(loc='lower right')

plt.subplot(3, 2, 4)
sns.lineplot(x=range(1, len(history_dict.get('recall', [])) + 1), y=history_dict.get('recall', []), label='Training Recall')
sns.lineplot(x=range(1, len(history_dict.get('val_recall', [])) + 1), y=history_dict.get('val_recall', []), label='Validation Recall')
plt.title('Model Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend(loc='lower right')

plt.subplot(3, 2, 5)
sns.lineplot(x=range(1, len(f1_scores) + 1), y=f1_scores, label='Validation F1 Score')
plt.title('Model F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()

# Save model
model.save('smnist_with_resnet.h5')
