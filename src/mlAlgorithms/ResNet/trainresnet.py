from keras.applications.resnet_v2 import ResNet152V2
from keras_preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.layers import Input
from os import path
import pandas as pd
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

dim_size = 112
weights = 'resnet_weights.h5'

df = pd.read_csv("train.csv")
datagen = ImageDataGenerator(
   rescale = 1./255,
   shear_range = 0.2,
   zoom_range = 0.2,
   horizontal_flip = True,
   validation_split = 0.1)
train_generator = datagen.flow_from_dataframe(
   dataframe = df, 
   directory = "train_images", 
   x_col = "id", y_col = "label",
   subset="training", 
   class_mode = "categorical", 
   target_size = (dim_size,dim_size), 
   batch_size = 32
)
valid_generator = datagen.flow_from_dataframe(
   dataframe = df, 
   directory = "train_images", 
   x_col = "id", y_col = "label",
   subset="validation", 
   class_mode = "categorical", 
   target_size = (dim_size,dim_size), 
   batch_size = 32
)

input_tensor = Input(shape=(dim_size, dim_size, 3))
model = ResNet152V2(input_tensor=input_tensor, weights=None, classes=7)

if path.exists(weights):
   model.load_weights('resnet_weights.h5', by_name=False)

for layer in model.layers:
    layer.trainable = True

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=["accuracy"])
#from keras.optimizers import SGD
#model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=35)

model.save_weights(weights)