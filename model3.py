import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import cv2

# Load data
train_df = pd.read_csv("sign_mnist_train.csv")
test_df = pd.read_csv("sign_mnist_test.csv")

y_train = train_df['label']
y_test = test_df['label']
x_train = train_df.drop(columns=['label']).values
x_test = test_df.drop(columns=['label']).values

label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.transform(y_test)

# Normalize and resize images
x_train = x_train / 255.0
x_test = x_test / 255.0

# Use a lower resolution if memory issues persist
x_train = np.array([cv2.resize(img.reshape(28, 28), (160, 160)) for img in x_train])
x_test = np.array([cv2.resize(img.reshape(28, 28), (160, 160)) for img in x_test])

# Convert grayscale images to RGB
x_train_rgb = np.repeat(x_train[..., np.newaxis], 3, axis=-1)
x_test_rgb = np.repeat(x_test[..., np.newaxis], 3, axis=-1)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1, 
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)
datagen.fit(x_train_rgb)

# Load pre-trained EfficientNetB0 model without top layers
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(160, 160, 3))
for layer in base_model.layers:
    layer.trainable = False

# Build the model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(len(label_binarizer.classes_), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', 
              metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])

# Custom callback for additional metrics
class MetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            y_pred = self.model.predict(x_test_rgb, batch_size=32)
            y_pred_classes = np.argmax(y_pred, axis=-1)
            y_true_classes = np.argmax(y_test, axis=-1)

            precision = precision_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
            recall = recall_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
            f1 = f1_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)

            logs['val_precision'] = precision
            logs['val_recall'] = recall
            logs['val_f1'] = f1

metrics_callback = MetricsCallback()

# Train the model
history = model.fit(datagen.flow(x_train_rgb, y_train, batch_size=32), 
                    epochs=10, 
                    validation_data=(x_test_rgb, y_test),
                    callbacks=[metrics_callback])

# Plot results
sns.set(style="whitegrid")

plt.figure(figsize=(20, 15))

plt.subplot(3, 2, 1)
sns.lineplot(x=range(1, len(history.history['loss']) + 1), y=history.history['loss'], label='Training Loss')
sns.lineplot(x=range(1, len(history.history['val_loss']) + 1), y=history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.subplot(3, 2, 2)
sns.lineplot(x=range(1, len(history.history['accuracy']) + 1), y=history.history['accuracy'], label='Training Accuracy')
sns.lineplot(x=range(1, len(history.history['val_accuracy']) + 1), y=history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.subplot(3, 2, 3)
sns.lineplot(x=range(1, len(history.history.get('precision', [])) + 1), y=history.history.get('precision', []), label='Training Precision')
sns.lineplot(x=range(1, len(history.history.get('val_precision', [])) + 1), y=history.history.get('val_precision', []), label='Validation Precision')
plt.title('Model Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend(loc='lower right')

plt.subplot(3, 2, 4)
sns.lineplot(x=range(1, len(history.history.get('recall', [])) + 1), y=history.history.get('recall', []), label='Training Recall')
sns.lineplot(x=range(1, len(history.history.get('val_recall', [])) + 1), y=history.history.get('val_recall', []), label='Validation Recall')
plt.title('Model Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend(loc='lower right')

plt.subplot(3, 2, 5)
f1_scores = history.history.get('val_f1', [])
sns.lineplot(x=range(1, len(f1_scores) + 1), y=f1_scores, label='Validation F1 Score')
plt.title('Model F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()

# Save model
model.save('smnist_efficientnet.h5')
