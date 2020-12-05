import numpy as np
import gzip
import struct

def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        _ignored, n_images, columns, rows = struct.unpack('>IIII', f.read(16))
        all_pixels = np.frombuffer(f.read(), dtype=np.uint8)
        return all_pixels.reshape(n_images, columns * rows)

def prepend_bias(X):
    return np.insert(X, 0, 1, axis=1)

X_train = prepend_bias(load_images("train-images-idx3-ubyte.gz"))
X_test = prepend_bias(load_images("t10k-images-idx3-ubyte.gz"))

def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(8)
        all_labels = f.read()
        return np.frombuffer(all_labels, dtype=np.uint8).reshape(-1,1)

def encode_fives(Y):
    return (Y == 7).astype(int)

Y_train = encode_fives(load_labels("train-labels-idx1-ubyte.gz"))

Y_test = encode_fives(load_labels("t10k-labels-idx1-ubyte.gz"))

