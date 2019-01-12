Cityscapes_images_dir = '../datasets/Cityscapes/JPEGImages'

# The directories that contain the annotations.
Cityscapes_annotation_dir = '../datasets/Cityscapes/Annotations'

# The paths to the image sets.
Cityscapes_train_source_image_set_filename = '../datasets/Cityscapes/ImageSets/Main/train_source.txt'
Cityscapes_train_target_image_set_filename = '../datasets/Cityscapes/ImageSets/Main/train_target.txt'
Cityscapes_test_target_image_set_filename = '../datasets/Cityscapes/ImageSets/Main/test.txt'

images_dirs = [Cityscapes_images_dir]
annotations_dirs = [Cityscapes_annotation_dir]
image_set_filenames = [Cityscapes_train_source_image_set_filename, Cityscapes_train_target_image_set_filename]


# Erase data that might have been parsed before.
filenames = []
image_ids = []
labels = []
eval_neutral = []


for images_dir, image_set_filename, annotations_dir in zip(images_dirs, image_set_filenames, annotations_dirs):
    # Read the image set file that so that we know all the IDs of all the images to be included in the dataset.
    with open(image_set_filename) as f:
        image_ids = [line.strip() for line in f]  # Note: These are strings, not integers.
        image_ids += image_ids

print('True')
