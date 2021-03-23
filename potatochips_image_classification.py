# driveファイルをマウントする
from google.colab import drive
drive.mount('/content/drive')

# drive/MyDriveにあるpotato-chips.zipを解凍する
!unzip "drive/MyDrive/potato-chips.zip"

# potato-chipsを入れるディレクトリを作る
!mkdir raw_data

# potato-chipsをraw_dataに入れる
!for d in $(ls -d potato-chips/* | grep "/"); do for f in $(ls -1F ${d}/* | xargs -i basename {}); do cp -f "${d}"/$f raw_data/"${d#*/}.${f}"; done; done

# 枚数を確認
!ls raw_data / * | wc - l
!for d in $(ls -d potato-chips / * | grep "/"); do for f in $(ls ${d} / * | wc -l);do echo "${d#*/}:${f}";done;done

# training_dataとtest_dataを消す。
!rm - rf
train_data
!rm - rf
test_data
# training_dataとtest_dataを作る
!mkdir
train_data
!mkdir
test_data

!cp - f
raw_data / * train_data
RAND_REPRO = False  # ここをFlaseにすると新たにランダム抽出する。Trueにするとtest_data_list.csvに記載されているファイルを抽出する。
if RAND_REPRO:
    !while read file_name; do mv./ train_data / ${file_name}./ test_data; done < Xception_version1_test.csv
else:
    !for f in $(ls -d potato-chips / * | grep "/"); do find./ train_data / | grep "${f#*/}.IMG" | sort -R | tail -n 20 | xargs -n 1 sh -c 'mv ${0} ./test_data';done
!rm - rf
test_data_list.csv
!ls - 1. / test_data
1 > test_data_list.csv
!ls - 1. / test_data

# train_dataとtest_dataの枚数を確認
!ls train_data/* | wc -l
!ls test_data/* | wc -l

# モジュールのインポート
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os

# 定数を定義
FAST_RUN = False
IMAGE_WIDTH=192
IMAGE_HEIGHT=256
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

# それぞれのデータにレベル付け
filenames = os.listdir("./train_data")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    categories.append(category)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

# dfの確認
df

# それぞれの枚数の確認
df['category'].value_counts().plot.bar()

# モデルに使うモジュールのインポート
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

# Xception用
from keras.applications.xception import Xception
from keras.layers.pooling import GlobalAveragePooling2D

# モデルの作成
xception=Xception(input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),include_top=False,weights='imagenet')
x = xception.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation = 'relu')(x)
predictions = Dense(7, activation = 'softmax')(x)
model = Model(inputs = xception.input, outputs = predictions)

# 転移モデル(Xception)の層を凍結する
for layer in model.layers:
    layer.trainable = False

    if layer.name.startswith('batch_normalization'):
        layer.trainable = True
    if layer.name.endswith('bn'):
        layer.trainable = True

 # optimizer、loss、metricsを指定
 model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ["accuracy"]
)

# modelを確認
model.summary()

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlystop = EarlyStopping(monitor = 'val_loss',
                          patience = 10,
                          verbose = 1)

# 学習率をどう調整するかを定義
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_loss',
                                            factor = 0.1,
                                            patience = 3,
                                            verbose = 1,
                                            min_lr=0.00001)
# callbacks関数を定義
callbacks = [earlystop, learning_rate_reduction]

# trainとvalidateを作る
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
train_df

# trainデータ、validateデータの種類ごとの割合を確認
train_df['category'].value_counts().plot.bar()
validate_df['category'].value_counts().plot.bar()

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=32

# Data Augmentation validateには適用しない
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    "./train_data/",
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    "./train_data/",
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

# 学習させる
epochs=3 if FAST_RUN else 50
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks,
    shuffle = True, # 違う
    verbose= 1
)

# 重みを保存
model.save_weights("xception_adam_model.h5")

# Visualize Training 
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()

# Preparing Test Data
test_filenames = os.listdir("./test_data")
test_labels = []
for test_filename in test_filenames:
  test_label = test_filename.split('.')[0]
  test_labels.append(test_label)

test_df = pd.DataFrame({
    'filename' : test_filenames,
    'label' : test_labels
})
nb_samples = test_df.shape[0]

# Generate Test Data
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df,
    "./test_data/",
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size, #batch_sizeが違う
    shuffle=False
)

# Predict
predict = model.predict(test_generator, steps=np.ceil(nb_samples/batch_size))

test_df['category'] = np.argmax(predict, axis=-1)
test_df

label_map = dict((v, k) for k, v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)

# VISUALIZE RESULT
test_df['category'].value_counts().plot.bar()

# Confusion matrix
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix, without normalization')

  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=90)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')

confusion_mtx = confusion_matrix(test_df['label'], test_df['category'])
plot_confusion_matrix(confusion_mtx, classes = range(7))

# Caliculate F1 score(Accracy)
from sklearn.metrics import f1_score
point = f1_score(test_df['label'], test_df['category'], average='micro')

# Convert to Percentage
print('accuracyは' + str(100 * point) + '%')

# submission
submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category']
submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv('submission.csv', index=False)