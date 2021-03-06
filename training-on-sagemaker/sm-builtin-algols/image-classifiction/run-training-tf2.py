import argparse, os
import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    print(os.environ['SM_CHANNEL_TRAINING'])
    print(os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    
    args, _ = parser.parse_known_args()
    
    epochs     = args.epochs
    lr         = args.learning_rate
    batch_size = args.batch_size
    gpu_count  = args.gpu_count
    model_dir  = args.model_dir
    training_dir   = args.training
    validation_dir = args.validation
    num_classes = 5
    image_resize = 150
    batch_size_training = 4
    batch_size_validation = 4
    data_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    )

    
    train_generator = data_generator.flow_from_directory(
    training_dir,
    target_size=(image_resize, image_resize),
    batch_size=batch_size_training,
    class_mode='categorical')
    
    ## and another one for the validation set.

    validation_generator = data_generator.flow_from_directory(
    validation_dir,
    target_size=(image_resize, image_resize),
    batch_size=batch_size_validation,
    class_mode='categorical')

    # Initialising Model
    model = Sequential()
    model.add(ResNet50(
            include_top=False,
            pooling='avg',
            weights='imagenet',
            ))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    model.layers[0].trainable = False
    
    print(model.summary())
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])   
    #     steps_per_epoch_training = len(train_generator) / batch_size_training
    #     steps_per_epoch_validation = len(validation_generator) / batch_size_validation
    num_epochs = 10

    fit_history = model.fit_generator(
                train_generator,
            #     steps_per_epoch=steps_per_epoch_training,
                epochs=num_epochs,
                validation_data=validation_generator,
            #     validation_steps=steps_per_epoch_validation,
                verbose=1,
            )

    #score = model.evaluate(x_val, y_val, verbose=0)
    score=model.evaluate_generator(validation_generator, steps=1, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
    print('Validation loss    :', score[0])
    print('Validation accuracy:', score[1])

#     model.save(os.path.join(model_dir, 'my_model.h5'))
#---> WORKING ONE     model.save(os.path.join(model_dir, 'model'))
    model_path = '{}/{}/00000001'.format(model_dir, 'my_model')
    model.save(model_path)

#     # save Keras model for Tensorflow Serving
#     sess = K.get_session()
#     tf.saved_model.simple_save(
#         sess,
#         os.path.join(model_dir, 'model/1'),
#         inputs={'inputs': model.input},
#         outputs={t.name: t for t in model.outputs})
    
#     model.save('{}/model/1/model.h5'.format(model_dir))
#     model.save(os.path.join(model_dir, 'model'))