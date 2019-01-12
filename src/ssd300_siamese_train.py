import sys
sys.path.append('../')
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import glob

from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd300_Siamese import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms


img_height = 300 # Height of the model input images
img_width = 300 # Width of the model input images
img_channels = 3 # Number of color channels of the model input images
mean_color = [123, 117, 104] # Per-channel mean of images. Do not change if use any of the pre-trained weights.
# The color channel order in the original SSD is BGR,
# so we'll have the model reverse the color channel order of the input images.
swap_channels = [2, 1, 0]
classes = ['background',
           'person', 'rider', 'car', 'truck',
           'bus', 'train', 'motorcycle', 'bicycle']
n_classes = len(classes) - 1  # Number of positive classes, 8 for domain Cityscapes, 20 for Pascal VOC, 80 for MS COCO
# The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
# scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
# The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
scales = scales_coco
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300] # The space between two adjacent anchor box center points for each predictor layer.
# The offsets of the first anchor box center points from the top and left borders of the image
# as a fraction of the step size for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
# The variances by which the encoded target coordinates are divided as in the original implementation
variances = [0.1, 0.1, 0.2, 0.2]
normalize_coords = True
Model_Build = 'New_Model'  # 'Load_Model'
Optimizer_Type = 'Adam'  # 'SGD'  #
batch_size = 16  # Change the batch size if you like, or if you run into GPU memory issues.
alpha_distance = 0.0001  # Coefficient for the distance between the source and target feature maps.

if len(glob.glob('*.h5')):
    Dataset_Build = 'Load_Dataset'
else:
    Dataset_Build = 'New_Dataset'

if Model_Build == 'New_Model':
    # 1: Build the Keras model.

    K.clear_session()  # Clear previous models from memory.

    # import tensorflow as tf
    # from keras.backend.tensorflow_backend import set_session
    #
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    # config.log_device_placement = True  # to log device placement (on which device the operation ran)
    # # (nothing gets printed in Jupyter, only if you run it standalone)
    # sess = tf.Session(config=config)
    # set_session(sess)  # set this TensorFlow session as the default session for Keras

    model = ssd_300(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='training',
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_per_layer=aspect_ratios,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords,
                    subtract_mean=mean_color,
                    swap_channels=swap_channels)

    # 2: Load some weights into the model.

    # TODO: Set the path to the weights you want to load.
    weights_path = '../trained_weights/VGG_ILSVRC_16_layers_fc_reduced.h5'

    model.load_weights(weights_path, by_name=True)

    # 3: Instantiate an optimizer and the SSD loss function and compile the model.
    #    If you want to follow the original Caffe implementation, use the preset SGD
    #    optimizer, otherwise I'd recommend the commented-out Adam optimizer.

    if Optimizer_Type == 'SGD':
        Optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
    elif Optimizer_Type == 'Adam':
        Optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    else:
        raise ValueError('Undefined Optimizer_Type.')

    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0, alpha_distance=alpha_distance)

    model.compile(optimizer=Optimizer, loss={'conv9_2_norm_substract': ssd_loss.compute_distance_loss,
                                             'predictions': ssd_loss.compute_loss})

elif Model_Build == 'Load_Model':
    # TODO: Set the path to the `.h5` file of the model to be loaded.
    model_path = '../trained_weights/VGG_ssd300_Cityscapes/epoch-23_loss-5.2110_val_loss-6.7452.h5'

    # We need to create an SSDLoss object in order to pass that to the model loader.
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0, alpha_distance=alpha_distance)

    K.clear_session()  # Clear previous models from memory.

    # import tensorflow as tf
    # from keras.backend.tensorflow_backend import set_session
    #
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    # config.log_device_placement = True  # to log device placement (on which device the operation ran)
    # # (nothing gets printed in Jupyter, only if you run it standalone)
    # sess = tf.Session(config=config)
    # set_session(sess)  # set this TensorFlow session as the default session for Keras

    model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                   'L2Normalization': L2Normalization,
                                                   'compute_loss': ssd_loss.compute_loss,
                                                   'compute_distance_loss': ssd_loss.compute_distance_loss})
else:
    raise ValueError('Undefined Model_Build. Model_Build should be New_Model  or Load_Model')

if Dataset_Build == 'New_Dataset':
    # 1: Instantiate two `DataGenerator` objects: One for training, one for validation.

    # Optional: If you have enough memory, consider loading the images into memory for the reasons explained above.

    train_dataset = DataGenerator(dataset='train', load_images_into_memory=False, hdf5_dataset_path=None)
    val_dataset = DataGenerator(dataset='val', load_images_into_memory=False, hdf5_dataset_path=None)

    # 2: Parse the image and label lists for the training and validation datasets. This can take a while.

    # TODO: Set the paths to the datasets here.

    # Introduction of PascalVOC: https://arleyzhang.github.io/articles/1dc20586/
    # The directories that contain the images.
    Cityscapes_images_dir = '../../datasets/Cityscapes/JPEGImages'

    # The directories that contain the annotations.
    Cityscapes_annotation_dir = '../../datasets/Cityscapes/Annotations'

    # The paths to the image sets.
    Cityscapes_train_source_image_set_filename = '../../datasets/Cityscapes/ImageSets/Main/train_source.txt'
    Cityscapes_train_target_image_set_filename = '../../datasets/Cityscapes/ImageSets/Main/train_target.txt'
    Cityscapes_test_target_image_set_filename = '../../datasets/Cityscapes/ImageSets/Main/test.txt'

    # images_dirs, image_set_filenames, and annotations_dirs should have the same length
    train_dataset.parse_xml(images_dirs=[Cityscapes_images_dir,
                                         Cityscapes_images_dir],
                            image_set_filenames=[Cityscapes_train_source_image_set_filename,
                                                 Cityscapes_train_target_image_set_filename],
                            annotations_dirs=[Cityscapes_annotation_dir,
                                              Cityscapes_annotation_dir],
                            classes=classes,
                            include_classes='all',
                            exclude_truncated=False,
                            exclude_difficult=False,
                            ret=False)

    val_dataset.parse_xml(images_dirs=[Cityscapes_images_dir],
                          image_set_filenames=[Cityscapes_test_target_image_set_filename],
                          annotations_dirs=[Cityscapes_annotation_dir],
                          classes=classes,
                          include_classes='all',
                          exclude_truncated=False,
                          exclude_difficult=True,
                          ret=False)

    # Optional: Convert the dataset into an HDF5 dataset. This will require more disk space, but will
    # speed up the training. Doing this is not relevant in case you activated the `load_images_into_memory`
    # option in the constructor, because in that cas the images are in memory already anyway. If you don't
    # want to create HDF5 datasets, comment out the subsequent two function calls.

    # After create these h5 files, if you have resized the input image, you need to reload these files. Otherwise,
    # the images and the labels will not change.

    resize_image_to = (300, 600)
    train_dataset.create_hdf5_dataset(file_path='dataset_cityscapes_train.h5',
                                      resize=resize_image_to,
                                      variable_image_size=True,
                                      verbose=True)

    val_dataset.create_hdf5_dataset(file_path='dataset_cityscapes_test.h5',
                                    resize=resize_image_to,
                                    variable_image_size=True,
                                    verbose=True)

    train_filename_paths = [Cityscapes_train_source_image_set_filename,
                            Cityscapes_train_target_image_set_filename]
    train_filenames = []
    for filename_path in train_filename_paths:
        with open(filename_path, 'r') as f:
            train_filenames.extend([os.path.join(Cityscapes_images_dir, line.strip()) for line in f])

    train_dataset = DataGenerator(dataset='train',
                                  load_images_into_memory=False,
                                  hdf5_dataset_path='dataset_cityscapes_train.h5',
                                  filenames=train_filenames,
                                  filenames_type='text',
                                  images_dir=Cityscapes_images_dir)

    val_dataset = DataGenerator(dataset='val',
                                load_images_into_memory=False,
                                hdf5_dataset_path='dataset_cityscapes_test.h5',
                                filenames=Cityscapes_test_target_image_set_filename,
                                filenames_type='text',
                                images_dir=Cityscapes_images_dir)

elif Dataset_Build == 'Load_Dataset':
    # 1: Instantiate two `DataGenerator` objects: One for training, one for validation.

    # Load dataset from the created h5 file.

    # The directories that contain the images.
    Cityscapes_images_dir = '../../datasets/Cityscapes/JPEGImages'

    # The paths to the image sets.
    Cityscapes_train_source_image_set_filename = '../../datasets/Cityscapes/ImageSets/Main/train_source.txt'
    Cityscapes_train_target_image_set_filename = '../../datasets/Cityscapes/ImageSets/Main/train_target.txt'
    Cityscapes_test_target_image_set_filename = '../../datasets/Cityscapes/ImageSets/Main/test.txt'

    train_filename_paths = [Cityscapes_train_source_image_set_filename,
                            Cityscapes_train_target_image_set_filename]
    train_filenames = []
    for filename_path in train_filename_paths:
        with open(filename_path, 'r') as f:
            train_filenames.extend([os.path.join(Cityscapes_images_dir, line.strip()) for line in f])

    train_dataset = DataGenerator(dataset='train',
                                  load_images_into_memory=False,
                                  hdf5_dataset_path='dataset_cityscapes_train.h5',
                                  filenames=train_filenames,
                                  filenames_type='text',
                                  images_dir=Cityscapes_images_dir)

    val_dataset = DataGenerator(dataset='val',
                                load_images_into_memory=False,
                                hdf5_dataset_path='dataset_cityscapes_test.h5',
                                filenames=Cityscapes_test_target_image_set_filename,
                                filenames_type='text',
                                images_dir=Cityscapes_images_dir)

else:
    raise ValueError('Undefined Dataset_Build. Dataset_Build should be New_Dataset or Load_Dataset.')

# 4: Set the image transformations for pre-processing and data augmentation options.

# For the training generator:
ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                            img_width=img_width,
                                            background=mean_color)

# For the validation generator:
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

# 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                   model.get_layer('fc7_mbox_conf').output_shape[1:3],
                   model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_per_layer=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.5,
                                    normalize_coords=normalize_coords)

# 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.
# The input image and label are first processed by transformations. Then, the label will be further encoded by
# ssd_input_encoder. The encoded labels are classId and offset to each anchor box.
train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=[ssd_data_augmentation],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[convert_to_3_channels,
                                                      resize],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

# while True:
#     batch_result = next(train_generator)



def lr_schedule(epoch):
    if epoch < 60:
        return 0.001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001


# Define model callbacks.
checkpoint_path = '../trained_weights/VGG_ssd300_Siamese_Cityscapes'
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

# TODO: Set the filepath under which you want to save the model.
model_checkpoint = ModelCheckpoint(filepath=os.path.join(checkpoint_path, 'epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5'),
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=False,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)

# model_checkpoint.best to the best validation loss from the previous training
# model_checkpoint.best = 4.83704

csv_logger = CSVLogger(filename=os.path.join(checkpoint_path, 'ssd300_Siamese_Cityscapes_training_log.csv'),
                       separator=',',
                       append=True)

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                verbose=1)

terminate_on_nan = TerminateOnNaN()

callbacks = [model_checkpoint,
             csv_logger,
             learning_rate_scheduler,
             terminate_on_nan]

initial_epoch = 0
final_epoch = 120
steps_per_epoch = 1000

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              callbacks=callbacks,
                              validation_data=val_generator,
                              validation_steps=ceil(val_dataset_size/batch_size),
                              initial_epoch=initial_epoch)

# 1: Set the generator for the val_dataset or train_dataset predictions.

predict_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=False,
                                         transformations=[convert_to_3_channels,
                                                          resize],
                                         label_encoder=None,
                                         returns={'processed_images',
                                                  'filenames',
                                                  'inverse_transform',
                                                  'original_images',
                                                  'original_labels'},
                                         keep_images_without_gt=False)

# 2: Generate samples.

batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(predict_generator)


i = 0  # Which batch item to look at

print("Image:", batch_filenames[i])
print()
print("Ground truth boxes:\n")
print(np.array(batch_original_labels[i]))

# 3: Make predictions.

y_pred = model.predict(batch_images)[-1]

# Now let's decode the raw predictions in `y_pred`.

# Had we created the model in 'inference' or 'inference_fast' mode,
# then the model's final layer would be a `DecodeDetections` layer and
# `y_pred` would already contain the decoded predictions,
# but since we created the model in 'training' mode,
# the model outputs raw predictions that still need to be decoded and filtered.
# This is what the `decode_detections()` function is for.
# It does exactly what the `DecodeDetections` layer would do,
# but using Numpy instead of TensorFlow (i.e. on the CPU instead of the GPU).

# `decode_detections()` with default argument values follows the procedure of the original SSD implementation:
# First, a very low confidence threshold of 0.01 is applied to filter out the majority of the predicted boxes,
# then greedy non-maximum suppression is performed per class with an intersection-over-union threshold of 0.45,
# and out of what is left after that, the top 200 highest confidence boxes are returned.
# Those settings are for precision-recall scoring purposes though.
# In order to get some usable final predictions, we'll set the confidence threshold much higher, e.g. to 0.5,
# since we're only interested in the very confident predictions.

# 4: Decode the raw predictions in `y_pred`.

y_pred_decoded = decode_detections(y_pred,
                                   confidence_thresh=0.35,
                                   iou_threshold=0.4,
                                   top_k=200,
                                   normalize_coords=normalize_coords,
                                   img_height=img_height,
                                   img_width=img_width)

# We made the predictions on the resized images,
# but we'd like to visualize the outcome on the original input images,
# so we'll convert the coordinates accordingly.
# Don't worry about that opaque `apply_inverse_transforms()` function below,
# in this simple case it just applies `(* original_image_size / resized_image_size)` to the box coordinates.

# 5: Convert the predictions for the original image.

y_pred_decoded_inv = apply_inverse_transforms(y_pred_decoded, batch_inverse_transforms)

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_decoded_inv[i])

# Finally, let's draw the predicted boxes onto the image.
# Each predicted box says its confidence next to the category name.
# The ground truth boxes are also drawn onto the image in green for comparison.

# 5: Draw the predicted boxes onto the image

# Set the colors for the bounding boxes
colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist()
classes = ['background',
           'person', 'rider', 'car', 'truck',
           'bus', 'train', 'motorcycle', 'bicycle']

plt.figure(figsize=(20, 12))
plt.imshow(batch_original_images[i])

current_axis = plt.gca()

for box in batch_original_labels[i]:
    xmin = box[1]
    ymin = box[2]
    xmax = box[3]
    ymax = box[4]
    label = '{}'.format(classes[int(box[0])])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': 'green', 'alpha': 1.0})

for box in y_pred_decoded_inv[i]:
    xmin = box[2]
    ymin = box[3]
    xmax = box[4]
    ymax = box[5]
    color = colors[int(box[0])]
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})
