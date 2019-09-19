from keras.applications.resnet_v2 import ResNet152V2
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd

df = pd.read_csv(r".\train.csv")
datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.2)
train_generator = datagen.flow_from_dataframe(
   dataframe = df, 
   directory = ".\train_images", 
   x_col = "id", y_col = "label",
   subset="training", 
   class_mode = "categorical", 
   target_size = (224,224), 
   batch_size = 32
)
valid_generator = datagen.flow_from_dataframe(
   dataframe = df, 
   directory = ".\train_images", 
   x_col = "id", y_col = "label",
   subset="validation", 
   class_mode = "categorical", 
   target_size = (224,224), 
   batch_size = 32
)

model = ResNet152V2(input_shape=(224,224,3), weights=None, classes=7)

for layer in model.layers:
    layer.trainable = True

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=["accuracy"])

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
#from keras.optimizers import SGD
#model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# train the model on the new data for a few epochs
STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10)

model.save_weights('resnet_weights.h5')