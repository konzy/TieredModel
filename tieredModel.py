import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe


def print_report():
    # TODO: print more verbose report
    # TODO: create graphs and stuff
    print("reporting")

caffe.set_device(0)
caffe.set_mode_gpu()

# Nets Inits
binary_net = caffe.Net('binary.prototxt', 'binary_snapshot.caffemodel', caffe.TEST)
english_net = caffe.Net('english.prototxt', 'english_snapshot.caffemodel', caffe.TEST)
kannada_net = caffe.Net('kannada.prototxt', 'kannada_snapshot.caffemodel', caffe.TEST)

english_test_image = np.array(Image.open('test_english.png'))
kannada_test_image = np.array(Image.open('test_english.png'))
test_images = [english_test_image, kannada_test_image]

# Database Init
# validation database digits job : /usr/share/digits/digits/jobs/20160517-024602-8aa8
# TODO: Test with entire lmdb database
# TODO: preprocess images
# transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# transformer.set_mean('data', np.load('python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
# transformer.set_transpose('data', (2,0,1))
# transformer.set_channel_swap('data', (2,1,0))
# transformer.set_raw_scale('data', 255.0)

# For each training example
# Classify English or Kannada
# If English pass through english_net
# Else Kannada pass through kannada_net
for test_image in test_images:
    binary_net.blobs['data'].data[...] = test_image
    binary_net.forward()
    binary_classification = binary_net.blobs['prob'].data

    if binary_classification == 0:
        binary_net.blobs['data'].data[...] = test_image
        english_net.forward()
        english_char_classification = english_net.blobs['prob'].data
        print(english_char_classification)
    else:
        kannada_net.blobs['data'].data[...] = test_image
        kannada_net.forward()
        kannada_char_classification = kannada_net.blobs['prob'].data
        print(kannada_char_classification)

print_report()