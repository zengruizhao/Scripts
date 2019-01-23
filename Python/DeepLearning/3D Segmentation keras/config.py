import os
config = dict()
config["image_shape"] = (256, 160, 48, 1)
config["batch_size"] = 1
config["n_epochs"] = 200
config["decay_learning_rate_every_x_epochs"] = 500
config["initial_learning_rate"] = 0.001     # 0.0001
config["learning_rate_drop"] = 1
config["validation_split"] = 1
config["train_data_path"] = '/media/zzr/Data/Task07_Pancreas/Game/preprocess/img'
config["train_seg_path"] = '/media/zzr/Data/Task07_Pancreas/Game/preprocess/label'
config["test_img_path"] = '/media/zzr/Data/Task07_Pancreas/TCIA/preprocess/img'
config["test_seg_path"] = '/media/zzr/Data/Task07_Pancreas/TCIA/preprocess/label'
config["check_point_path"] = os.path.abspath("model")
config["check_point_name"] = os.path.abspath("check_point.hdf5")
config["model_file"] = "ruxian_3d_unet_model.h5"
config['smooth'] = 1.
config['csvLog'] = '/home/zzr/Data/pancreas/script/log/training_scrach2.log'
config['checkPoint'] = '/home/zzr/Data/pancreas/script/models'
config['saveCheckPointPerEpoch'] = 20
config['tensorBoard'] = '/home/zzr/Data/pancreas/script/tensorboard'
config['modelSave'] = 'pancreas_3d_unet_model.h5'

