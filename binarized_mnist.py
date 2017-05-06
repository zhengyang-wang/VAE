import numpy as np

def lines_to_np_array(lines):
    return np.array([[int(i) for i in line.split()] for line in lines])

class DataLoader(object):
    """ an object that generates batches of binarized MNIST data for training """

    def __init__(self, data_dir, subset, batch_size=100, rng=None, shuffle=False):
        """ 
        - data_dir is location where to store files
        - subset is train|valid|test
        - batch_size is int, of #examples to load at once
        - rng is np.random.RandomState object for reproducibility
        """

        self.batch_size = batch_size
        self.shuffle = shuffle

        data_file = open(data_dir + '/binarized_mnist_' + subset +'.amat', 'r')
        self.images = lines_to_np_array(data_file.readlines()).astype('float32')
        
        self.p = 0 # pointer to where we are in iteration
        self.rng = np.random.RandomState(1) if rng is None else rng

    def reset(self):
        self.p = 0

    def __iter__(self):
        return self

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        if n is None:
            n = self.batch_size

        # on first iteration lazily permute all data
        if self.p == 0 and self.shuffle:
            inds = self.rng.permutation(self.images.shape[0])
            self.images = self.images[inds]

        # on last iteration reset the counter and raise StopIteration
        if self.p + n > self.images.shape[0]:
            self.reset() # reset for next time we get called
            raise StopIteration

        # on intermediate iterations fetch the next batch
        x = self.images[self.p : self.p + n]
        self.p += self.batch_size

        return x

    next = __next__
