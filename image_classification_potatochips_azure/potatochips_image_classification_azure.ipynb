# Import packages
% matplotlib
inline
import numpy as np
import matplotlib.pyplot as plt

import azureml.core
from azureml.core import Workspace

print("Azure ML SDK Version: ", azureml.core.VERSION)

# %%

# Connect to a workspace
ws = Workspace.from_config()
print(ws.name, ws.location, ws.resource_group, sep='\t')

# %%

# Create an experiment
from azureml.core import Experiment

experiment_name = 'Potato-chips-classification'

exp = Experiment(workspace=ws, name=experiment_name)

# %%

# Attach an existing compute target
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
import os

compute_name = os.environ.get("AML_COMPUTE_CLUSTER_NAME", "cpu-cluster-C6V1")
compute_min_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MIN_NODES", 0)
compute_max_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MAX_NODES", 4)

vm_size = os.environ.get("AML_COMPUTE_CLUSTER_SKU", "STANDARD_NC6S_V3")

if compute_name in ws.compute_targets:
    compute_target = ws.compute_targets[compute_name]
    if compute_target and type(compute_target) is AmlCompute:
        print('found compute target. just use it. ' + compute_name)
else:
    print('creating a new compute target...')
    provisioning_config = AmlCompute.provisioning_configuration(vm_size=vm_size,
                                                                min_nodes=compute_min_nodes,
                                                                max_nodes=compute_max_nodes)

    compute_target = ComputeTarget.create(
        ws, compute_name, provisioning_config)

    compute_target.wait_for_completion(
        show_output=True, min_node_count=None, timeout_in_minutes=20)

    print(compute_target_C6V1.get_status().serialize())

# %%

# Create a directory
import os

data_folder = os.path.join(os.getcwd(), "data")
os.makedirs(data_folder, exist_ok=True)
script_folder = os.path.join(os.getcwd(), "potato-chips-script")
os.makedirs(script_folder, exist_ok=True)

# %%


# Download Dataset
# azureml-core of version 1.0.72 or higher is required
from azureml.core import Workspace, Dataset

subscription_id = '0c3a9701-fb22-41b3-b904-89b9960d70d6'
resource_group = 'Koichi_1'
workspace_name = 'koichi_1'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='potatochips-data-dataset')
# dataset.download(target_path='.', overwrite=False)


# %%

% % writefile $script_folder / train.py

# Create a training script
import argparse
import os
import numpy as np

from azureml.core import Run

import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--optimizer', type=str, dest='arg_optimizer', help='optimizer')
parser.add_argument('--monitor', type=str, dest='arg_monitor', help='Learning_rate_reduction_monitor')
parser.add_argument('--batchsize', type=int, dest='arg_batchsize', help='batchsize')
parser.add_argument('--epochs', type=int, dest='arg_epochs', help='epochs')
parser.add_argument('--Learning_rate', type=float, dest='arg_Learning_rate', help='Learning_rate')
args = parser.parse_args()

arg_optimizer = args.arg_optimizer
arg_monitor = args.arg_monitor
arg_batchsize = args.arg_batchsize
arg_epochs = args.arg_epochs
arg_Learning_rate = args.arg_Learning_rate

data_folder = args.data_folder
print('Data folder:', data_folder)

training_folder = os.path.join(data_folder, "potato-chips")
print('Training folder:', training_folder)

if arg_optimizer == "SGD":
    optimizer = optimizers.SGD(lr=arg_Learning_rate)
elif arg_optimizer == "RMSprop":
    optimizer = optimizers.RMSprop(lr=arg_Learning_rate)
elif arg_optimizer == "Adam":
    optimizer = optimizers.Adam(lr=arg_Learning_rate)
else:
    optimizer = 'rmsprop'

FAST_RUN = False
IMAGE_WIDTH = 192
IMAGE_HEIGHT = 256
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

filenames = os.listdir(training_folder)
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    categories.append(category)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor=arg_monitor,
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

callbacks = [learning_rate_reduction]

train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size = arg_batchsize

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    training_folder,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size,
    validate_filenames=False
)

validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    training_folder,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size,
    validate_filenames=False
)

run = Run.get_context()

epochs = 3 if FAST_RUN else arg_epochs
history = model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate // batch_size,
    steps_per_epoch=total_train // batch_size,
    callbacks=callbacks
)

# create log
run.log('optimizer', arg_optimizer)
run.log('monitor', arg_monitor)
run.log('batchsize', arg_batchsize)
run.log('epochs', arg_epochs)
run.log('accuracy', history.history['accuracy'][-1])
run.log('val_accuracy', history.history['val_accuracy'][-1])
run.log('loss', history.history['loss'][-1])
run.log('val_loss', history.history['val_loss'][-1])

val_accuracy = history.history['val_accuracy'][-1]

from azureml.core.run import Run

run_logger = Run.get_context()
run_logger.log("accuracy", float(val_accuracy))

output_folder = os.path.join(os.getcwd(), "outputs")
os.makedirs(output_folder, exist_ok=True)

model.save_weights(os.path.join(os.getcwd(), "model.h5"))

# %%

# print(history.history)

# %%

import shutil

shutil.copy('utils.py', script_folder)

# %%
# Configure the training job
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies

env = Environment('potato-env')
cd = CondaDependencies.create(pip_packages=['azureml-defaults', 'tensorflow', 'matplotlib'],
                              conda_packages=['scikit-learn==0.22.1'])

env.python.conda_dependencies = cd
env.register(workspace=ws)

# %%

# Create random sampling for hyperpearameter tuning
from azureml.train.hyperdrive import RandomParameterSampling
from azureml.train.hyperdrive import choice, loguniform

param_sampling = RandomParameterSampling({
    "--optimizer": choice('SGD', 'RMSprop', 'Adam'),
    "--monitor": choice('val_accuracy', 'val_loss'),
    "--batchsize": choice(16, 32, 64),
    "--epochs": choice(30, 50, 70, 90, 120, 150),
    "--Learning_rate": loguniform(-4, -1)
})

# %%

# Specify primary metric
from azureml.train.hyperdrive import PrimaryMetricGoal

primary_metric_name = "accuracy",
primary_metric_goal = PrimaryMetricGoal.MAXIMIZE

# %%

# from azureml.core.run import Run
# run_logger = Run.get_context()
# run_logger.log("accuracy", float(val_accuracy))

# %%

# BanditPolicy
from azureml.train.hyperdrive import BanditPolicy

early_termination_policy = BanditPolicy(slack_factor=0.1, evaluation_interval=1, delay_evaluation=5)

# %%

# Configure hyperparameter tuning
from azureml.core import ScriptRunConfig

args = ['--data-folder', dataset.as_mount()]

src = ScriptRunConfig(source_directory=script_folder,
                      script='train.py',
                      arguments=args,
                      compute_target=compute_target,
                      environment=env)

# %%

from azureml.train.hyperdrive import HyperDriveConfig

hd_config = HyperDriveConfig(run_config=src,
                             hyperparameter_sampling=param_sampling,
                             policy=early_termination_policy,
                             primary_metric_name='accuracy',
                             primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                             max_total_runs=100,
                             max_concurrent_runs=4)

# %%

from azureml.core.experiment import Experiment

experiment = Experiment(workspace, experiment_name)
hyperdrive_run = experiment.submit(hd_config)

# %%

# Jupyter widget
from azureml.widgets import RunDetails

RunDetails(hyperdrive_run).show()