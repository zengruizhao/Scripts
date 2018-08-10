import os


config = dict()
config["image_shape"] = (256, 160, 48)#(144,144,80)  # because it's shape is arbitrarily
config["n_labels"] = 1  # not including background
config["batch_size"] = 1
config["n_epochs"] = 200
config["decay_learning_rate_every_x_epochs"] = 50
config["initial_learning_rate"] = 0.001#0.0001
config["learning_rate_drop"] = 1
config["validation_split"] = 1
config["train_data_path"] = '/home/zzr/Data/pancreas/overall_train_test_7_3/train_img'
config["train_seg_path"] = '/home/zzr/Data/pancreas/overall_train_test_7_3/train_mask'
config["resume"] = False
config["check_point_path"] = os.path.abspath("model")
config["check_point_name"] = os.path.abspath("check_point.hdf5")
config["model_file"] = "ruxian_3d_unet_model.h5"
config["smooth"] = 1.0# dice loss
config["input_shape"] = None

config["deconvolution"] = False  # use deconvolution instead of up-sampling. Requires keras-contrib.
config["pool_size"] = (2, 2, 2)
# divide the number of filters used by by a given factor. This will reduce memory consumption.
config["downsize_nb_filters_factor"] = 1
