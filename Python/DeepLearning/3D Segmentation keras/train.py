import math
import os
from functools import partial

from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, Callback, LearningRateScheduler, EarlyStopping, TensorBoard
from keras.models import load_model

from generator import get_training_and_testing_generators
from utils import pickle_dump
from config import config
from model import light_resnet, deconv_conv_unet_model_3d_coordconv_gn_deep, best_shallow, Nest_Net
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.utils.vis_utils import plot_model

KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))

K.set_image_dim_ordering('tf')


# learning rate schedule
def step_decay(epoch, initial_lrate=config["initial_learning_rate"], drop=config["learning_rate_drop"],
               epochs_drop=config["decay_learning_rate_every_x_epochs"]):
    print initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))


class SaveLossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        pickle_dump(self.losses, "loss_history.pkl")


def get_callbacks(model_file):
    # model_checkpoint = ModelCheckpoint(filepath=model_file, save_best_only=True)
    model_checkpoint = ModelCheckpoint(filepath=os.path.join('/home/zzr/Data/pancreas/script/models/',
                                                             'pancreas_'+'weights.{epoch:03d}.h5'), period=20)
    logger = CSVLogger("/home/zzr/Data/pancreas/script/log/training_scrach1.log")
    # history = SaveLossHistory()
    scheduler = LearningRateScheduler(partial(step_decay,
                                              initial_lrate=config["initial_learning_rate"],
                                              drop=config["learning_rate_drop"],
                                              epochs_drop=config["decay_learning_rate_every_x_epochs"]))
    # earlystoping = EarlyStopping()
    # tensorboard = TensorBoard('./tensorboard/logs', histogram_freq=0, write_graph=True)
    return [model_checkpoint, logger, scheduler]


def load_old_model(model_file):
    print "Loading pre-trained model"
    return load_model(model_file,
                      custom_objects={'dice_coef_loss': dice_coef_loss,
                                      'dice_coef': dice_coef})


def main():
    check_point_path = os.path.join(os.getcwd(), "models")
    check_point = os.path.join(check_point_path, "ruxian_weights.099.h5")
    model_file = os.path.join(check_point_path, 'pancreas_3d_unet_model.h5')
    if not os.path.exists(check_point_path):
        os.makedirs(check_point_path)

    if False:
        model = load_old_model(check_point)
    else:
        model = Nest_Net(shape=(256, 160, 48, 1), classes=1)
    model.summary()     # print model
    plot_model(model, to_file='model.png')
    # model.load_weights('/home/zzr/Data/pancreas/script/models/pancreas_weights.060.h5')
    # get training and testing generators
    train_generator, nb_train_samples, testing_generator, \
        nb_testing_samples = get_training_and_testing_generators(train_data_path=config["train_data_path"],
                                                                 train_seg_path=config["train_seg_path"],
                                                                 batch_size=config["batch_size"])
    print('nb_testing_samples', nb_testing_samples)

    # run training
    train_model(model, model_file, check_point, train_generator, nb_train_samples, testing_generator, nb_testing_samples)


def train_model(model, model_file, check_point, training_generator, nb_training_samples, testing_generator, nb_testing_samples):
    model.fit_generator(generator=training_generator,
                        steps_per_epoch=nb_training_samples,
                        epochs=config["n_epochs"],
                        callbacks=get_callbacks(check_point),
                        validation_data=testing_generator,
                        validation_steps=nb_testing_samples
                        )
    # class_weight samples, should modify generator
    # classweights = [0.5134, 19.1375]
    # model.fit_generator(generator=training_generator,
    #                     steps_per_epoch=nb_trainin`g_samples,
    #                     epochs=config["n_epochs"],
    #                     callbacks=get_callbacks(check_point),
    #                     validation_data=testing_generator,
    #                     validation_steps=nb_testing_samples,
    #                     class_weight=classweights
    #                     )
    model.save_weights(model_file)


if __name__ == "__main__":
    main()