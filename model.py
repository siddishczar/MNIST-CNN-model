import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

train_df = pd.read_csv("sign_mnist_train.csv")
test_df = pd.read_csv("sign_mnist_test.csv")

y_train = train_df['label']
y_test = test_df['label']
del train_df['label']
del test_df['label']

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.fit_transform(y_test)

x_train = train_df.values
x_test = test_df.values

x_train = x_train / 255
x_test = x_test / 255

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False, 
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range = 0.1, 
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False)

datagen.fit(x_train)


model = Sequential()
model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Flatten())
model.add(Dense(units = 512 , activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(units = 24 , activation = 'softmax'))

model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy', tensorflow.keras.metrics.Precision(name='precision'), tensorflow.keras.metrics.Recall(name='recall')])
model.summary()

class MetricsCallback(tensorflow.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:

            y_pred = self.model.predict(x_test)
            y_pred_classes = y_pred.argmax(axis=-1)
            y_true_classes = y_test.argmax(axis=-1)
            

            precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
            recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
            f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
            

            logs['val_precision'] = precision
            logs['val_recall'] = recall
            logs['val_f1'] = f1

metrics_callback = MetricsCallback()

history = model.fit(datagen.flow(x_train,y_train, batch_size = 128) ,epochs = 20 , validation_data = (x_test, y_test))

def compute_f1_score(y_true, y_pred):
    y_pred_classes = y_pred.argmax(axis=-1)
    y_true_classes = y_true.argmax(axis=-1)
    return f1_score(y_true_classes, y_pred_classes, average='weighted')


history_dict = history.history


y_pred = model.predict(x_test)
f1_scores = [compute_f1_score(y_test, y_pred) for _ in range(len(history_dict['accuracy']))]

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
sns.lineplot(x=range(1, len(history_dict['precision']) + 1), y=history_dict['precision'], label='Training Precision')
sns.lineplot(x=range(1, len(history_dict['val_precision']) + 1), y=history_dict['val_precision'], label='Validation Precision')
plt.title('Model Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend(loc='lower right')

plt.subplot(3, 2, 4)
sns.lineplot(x=range(1, len(history_dict['recall']) + 1), y=history_dict['recall'], label='Training Recall')
sns.lineplot(x=range(1, len(history_dict['val_recall']) + 1), y=history_dict['val_recall'], label='Validation Recall')
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

model.save('smnist.h5')