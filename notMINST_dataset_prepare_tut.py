# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function

import hashlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

# methods to download noMINST from google servers
url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None


def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 5% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent


def maybe_download(filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    if force or not os.path.exists(filename):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


# download train and test set
train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

# Extract the dataset from the compressed .tar.gz file. This should give you a set of directories, labelled A through J.
num_classes = 10
np.random.seed(133)


def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall()
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    print(data_folders)
    return data_folders


train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

# test some of the images (problem 1)
# f = plt.figure()
# labels = ['A','B','C','D','E','F','G','H']
# np.random.seed()
# for n in range(8):
#     train_files = os.listdir(train_folders[n])
#     f.add_subplot(2, 4, n+1)
#     i = int(np.random.rand()*len(train_files))
#     plt.imshow(plt.imread(os.path.join(train_folders[n], train_files[i])), cmap='Greys')
#     plt.title(labels[n])
# plt.show()

# load up the dataset into a more manageable format
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.


def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) -
                          pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)


# test some of the images from the pickled datset (problem 2 and 3)
# f = plt.figure()
# classSize = []
# labels = ['A','B','C','D','E','F','G','H','I','J']
# np.random.seed()
# for n in range(10):
#     classData = pickle.load(open(train_datasets[n], "rb"))
#     f.add_subplot(2, 5, n+1)
#     classSize.append(classData.shape[0])
#     i = int(np.random.rand()*classData.shape[0])
#     plt.imshow(classData[i,:,:], cmap='Greys')
#     plt.title(labels[n])
# # plot the class sizes to ensure data is balanced across classes
# p = plt.figure()
# axes = plt.subplot()
# axes.plot(classSize)
# axes.set_ylim([52908,52914])
# plt.title("class sizes")
# plt.show()

# merge and prune data as needed (memory requirements to be evaluated)


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels


train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


# Randomize the data
def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

# make sure the data is still valid after shuffling (problem 4)
# assuming labelling is in order eg. A=1, B=2, C=3, ...
# pick some random examples in the trainin gset and ensire the labels are correct
# f = plt.figure()
# labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
# np.random.seed()
# for n in range(10):
#     f.add_subplot(2, 5, n + 1)
#     i = int(np.random.rand() * train_dataset.shape[0])
#     plt.imshow(train_dataset[i, :, :], cmap='Greys')
#     plt.title(labels[train_labels[i]])
# plt.show()

# save the data
pickle_file = 'notMNIST.pickle'
try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

# check data base for overlapping images between the training and testing dataset (problem 5)
# solution from stmax82 uses sha1 hashes to compare images it also creates a sanitised subset with no overlaps
# scipy.signal.correlate2d can be used to compare images that are approximately equal
train_hashes = [hashlib.sha1(x).digest() for x in train_dataset]
valid_hashes = [hashlib.sha1(x).digest() for x in valid_dataset]
test_hashes  = [hashlib.sha1(x).digest() for x in test_dataset]
# np.in1d returns boolean array of len(arg1) to indicate which elemetns of arg1 exists in arg2
valid_in_train = np.in1d(valid_hashes, train_hashes)
test_in_train  = np.in1d(test_hashes,  train_hashes)
test_in_valid  = np.in1d(test_hashes,  valid_hashes)

valid_keep = ~valid_in_train
test_keep  = ~(test_in_train | test_in_valid)

valid_dataset_clean = valid_dataset[valid_keep]
valid_labels_clean  = valid_labels [valid_keep]

test_dataset_clean = test_dataset[test_keep]
test_labels_clean  = test_labels [test_keep]

print("valid -> train overlap: %d samples" % valid_in_train.sum())
print("test  -> train overlap: %d samples" % test_in_train.sum())
print("test  -> valid overlap: %d samples" % test_in_valid.sum())

# use logistic classifier to train simple model (problem 6)
logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
logreg.fit(np.vstack([n.ravel() for n in train_dataset]), train_labels)
print("training score : %.3f (%s)" % (logreg.score(np.vstack([n.ravel() for n in test_dataset]), test_labels), '20k training set'))
logreg.fit(np.vstack([n.ravel() for n in train_dataset[:5000,:,:]]), train_labels[:5000])
print("training score : %.3f (%s)" % (logreg.score(np.vstack([n.ravel() for n in test_dataset]), test_labels), '5k training set'))
logreg.fit(np.vstack([n.ravel() for n in train_dataset[:1000,:,:]]), train_labels[:1000])
print("training score : %.3f (%s)" % (logreg.score(np.vstack([n.ravel() for n in test_dataset]), test_labels), '1k training set'))
logreg.fit(np.vstack([n.ravel() for n in train_dataset[:100,:,:]]), train_labels[:100])
print("training score : %.3f (%s)" % (logreg.score(np.vstack([n.ravel() for n in test_dataset]), test_labels), '100 training set'))
logreg.fit(np.vstack([n.ravel() for n in train_dataset[:50,:,:]]), train_labels[:50])
print("training score : %.3f (%s)" % (logreg.score(np.vstack([n.ravel() for n in test_dataset]), test_labels), '50 training set'))
