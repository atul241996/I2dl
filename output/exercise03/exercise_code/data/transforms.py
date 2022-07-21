"""
Definition of image-specific transform classes
"""

# pylint: disable=too-few-public-methods

import numpy as np


class RescaleTransform:
    """Transform class to rescale images to a given range"""
    def __init__(self, out_range=(0, 1), in_range=(0, 255)):
        """
        :param out_range: Value range to which images should be rescaled to
        :param in_range: Old value range of the images
            e.g. (0, 255) for images with raw pixel values
        """
        self.min = out_range[0]
        self.max = out_range[1]
        self._data_min = in_range[0]
        self._data_max = in_range[1]

    def __call__(self, images):
        ########################################################################
        # TODO:                                                                #
        # Rescale the given images:                                            #
        #   - from (self._data_min, self._data_max)                            #
        #   - to (self.min, self.max)                                          #
        ########################################################################
        #images = images.resize(images, (self.min, self.max))
#         print(type(images))# <class 'numpy.ndarray'>
        #print(images.dtype)# float64
        #print(images.size) # 3072
        #print(images.shape)#32 32 3
        #print(dir(images))
        #images_rescaled = Image.fromarray(images)
        #print("images_rescaled"+images_rescaled)
        #new_size = self.min, 
        #new_ima=np.array(np.resize(images,))
        images = np.divide((images-self._data_min),(self._data_min-self._data_max))
        #print(new_ima)
        #print(images)
        #print("new_ima"+new_ima)
#         images.resize((self.min,self.max),refcheck=False)
        #test1=np.reshape(images,(self.min,self.max))
        #print (test1)
        #print("test1 is above ")
        #print (images.mode)
        #print("test1 is above ")
        #print (images.size)
        
        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return images
    

def compute_image_mean_and_std(images):
    """
    Calculate the per-channel image mean and standard deviation of given images
    :param images: numpy array of shape NxHxWxC
        (for N images with C channels of spatial size HxW)
    :returns: per-channels mean and std; numpy array of shape C
    """
    mean, std = None, None
    ########################################################################
    # TODO:                                                                #
    # Calculate the per-channel mean and standard deviation of the images  #
    # Hint: You can use numpy to calculate the mean and standard deviation #
    ########################################################################
    
    #print(images.shape)
    
    #val = np.reshape(images[:,:,0], -1)
    #print(val)
#     std0 = np.std(images[:,:,:,0])
#     std1 = np.std(images[:,:,:,1])
#     std2 = np.std(images[:,:,:,2])
#     mean0 = np.mean(images[:,:,:,0])
#     mean1 = np.mean(images[:,:,:,1])
#     mean2 = np.mean(images[:,:,:,2])
    #mean = images.mean(axis=(0,1,2))
    #std = images.std(axis=(0,1,2))
    mean = np.mean(images, axis=(0,1,2))
    std = np.std(images, axis=(0,1,2))
    
    pass

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return mean, std


class NormalizeTransform:
    """
    Transform class to normalize images using mean and std
    Functionality depends on the mean and std provided in __init__():
        - if mean and std are single values, normalize the entire image
        - if mean and std are numpy arrays of size C for C image channels,
            then normalize each image channel separately
    """
    def __init__(self, mean, std):
        """
        :param mean: mean of images to be normalized
            can be a single value, or a numpy array of size C
        :param std: standard deviation of images to be normalized
             can be a single value or a numpy array of size C
        """
        self.mean = mean
        self.std = std

    def __call__(self, images):
        ########################################################################
        # TODO:                                                                #
        # normalize the given images:                                          #
        #   - substract the mean of dataset                                    #
        #   - divide by standard deviation                                     #
        ########################################################################
        #print(images.shape)
        #print(self.mean)
        #print(self.std)
        images[:,:, 0] /= self.std[0]
        images[:,:, 1] /= self.std[1]
        images[:,:, 2] /= self.std[2]
        images[:,:, 0] -= self.mean[0]
        images[:,:, 1] -= self.mean[1]
        images[:,:, 2] -= self.mean[2]
        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return images


class ComposeTransform:
    """Transform class that combines multiple other transforms into one"""
    def __init__(self, transforms):
        """
        :param transforms: transforms to be combined
        """
        self.transforms = transforms

    def __call__(self, images):
        for transform in self.transforms:
            images = transform(images)
        return images
