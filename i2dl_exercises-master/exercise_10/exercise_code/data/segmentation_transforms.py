from abc import ABC, abstractmethod
import math
import numbers
import random

import numpy as np
from PIL import Image, PILLOW_VERSION
import torch
import torchvision
import torchvision.transforms.functional as F

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


class SegmentationTransform(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, image: Image, target: Image):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class SegmentationCompose(SegmentationTransform):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def __call__(self, image: Image, target: Image):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class SegmentationResize(SegmentationTransform):
    def __init__(self, size, interpolation=Image.BILINEAR):
        super().__init__()
        self.resize_transform = torchvision.transforms.Resize(size, interpolation)
        # ONLY ALLOWED INTERPOLATION METHOD FOR THE TARGET
        target_interpolation = Image.NEAREST
        self.target_transform = torchvision.transforms.Resize(size, target_interpolation)

    def __call__(self, image: Image, target: Image):
        image = self.resize_transform(image)
        target = self.target_transform(target)
        return image, target

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.resize_transform.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(
            self.resize_transform.size, interpolate_str)


class SegmentationToTensor(SegmentationTransform):
    def __init__(self):
        super().__init__()
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, image: Image, target: Image):
        image = self.to_tensor(image)
        target = self.to_tensor(target)
        return image, target

    def __repr__(self):
        return self.__class__.__name__ + '()'


class SegmentationRandomHorizontalFlip(SegmentationTransform):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, image: Image, target: Image):
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class SegmentationColorJitter(SegmentationTransform):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super(SegmentationColorJitter, self).__init__()
        self.color_jitter_transform = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image: Image, target: Image):
        image = self.color_jitter_transform(image)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.color_jitter_transform.brightness)
        format_string += ', contrast={0}'.format(self.color_jitter_transform.contrast)
        format_string += ', saturation={0}'.format(self.color_jitter_transform.saturation)
        format_string += ', hue={0})'.format(self.color_jitter_transform.hue)
        return format_string


class SegmentationRandomAffine(SegmentationTransform):
    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=Image.BILINEAR, fillcolor=0):
        super().__init__()
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and \
                    (len(shear) == 2 or len(shear) == 4), \
                    "shear should be a list or tuple and it must be of length 2 or 4."
                # X-Axis shear with [min, max]
                if len(shear) == 2:
                    self.shear = [shear[0], shear[1], 0., 0.]
                elif len(shear) == 4:
                    self.shear = [s for s in shear]
        else:
            self.shear = shear

        # ONLY ALLOWED INTERPOLATION METHOD
        self.resample = resample
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            if len(shears) == 2:
                shear = [random.uniform(shears[0], shears[1]), 0.]
            elif len(shears) == 4:
                shear = [random.uniform(shears[0], shears[1]),
                         random.uniform(shears[2], shears[3])]
        else:
            shear = 0.0

        return angle, translations, scale, shear

    @staticmethod
    def _get_inverse_affine_matrix(center, angle, translate, scale, shear):
        # Helper method to compute inverse matrix for affine transformation

        # As it is explained in PIL.Image.rotate
        # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        #       RSS is rotation with scale and shear matrix
        #       RSS(a, s, (sx, sy)) =
        #       = R(a) * S(s) * SHy(sy) * SHx(sx)
        #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(x)/cos(y) - sin(a)), 0 ]
        #         [ s*sin(a + sy)/cos(sy), s*(-sin(a - sy)*tan(x)/cos(y) + cos(a)), 0 ]
        #         [ 0                    , 0                                      , 1 ]
        #
        # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
        # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
        #          [0, 1      ]              [-tan(s), 1]
        #
        # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

        if isinstance(shear, numbers.Number):
            shear = [shear, 0]

        if not isinstance(shear, (tuple, list)) and len(shear) == 2:
            raise ValueError(
                "Shear should be a single value or a tuple/list containing " +
                "two values. Got {}".format(shear))

        rot = math.radians(angle)
        sx, sy = [math.radians(s) for s in shear]

        cx, cy = center
        tx, ty = translate

        # RSS without scaling
        a = np.cos(rot - sy) / np.cos(sy)
        b = -np.cos(rot - sy) * np.tan(sx) / np.cos(sy) - np.sin(rot)
        c = np.sin(rot - sy) / np.cos(sy)
        d = -np.sin(rot - sy) * np.tan(sx) / np.cos(sy) + np.cos(rot)

        # Inverted rotation matrix with scale and shear
        # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
        M = [d, -b, 0,
             -c, a, 0]
        M = [x / scale for x in M]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        M[2] += M[0] * (-cx - tx) + M[1] * (-cy - ty)
        M[5] += M[3] * (-cx - tx) + M[4] * (-cy - ty)

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        M[2] += cx
        M[5] += cy
        return M

    @staticmethod
    def affine(img, angle, translate, scale, shear, resample=0, fillcolor=None):
        assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
            "Argument translate should be a list or tuple of length 2"
        assert scale > 0.0, "Argument scale should be positive"

        output_size = img.size
        center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)
        matrix = SegmentationRandomAffine._get_inverse_affine_matrix(center, angle, translate, scale, shear)
        kwargs = {"fillcolor": fillcolor} if PILLOW_VERSION[0] >= '5' else {}
        return img.transform(output_size, Image.AFFINE, matrix, resample, **kwargs)

    def __call__(self, image: Image, target: Image):
        angle, translations, scale, shear = self.get_params(self.degrees, self.translate, self.scale, self.shear, image.size)
        image = SegmentationRandomAffine.affine(image, angle, translations, scale, shear, resample=self.resample, fillcolor=self.fillcolor)
        target = SegmentationRandomAffine.affine(target, angle, translations, scale, shear, resample=Image.NEAREST, fillcolor=self.fillcolor)
        return image, target

    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        if self.resample > 0:
            s += ', resample={resample}'
        if self.fillcolor != 0:
            s += ', fillcolor={fillcolor}'
        s += ')'
        d = dict(self.__dict__)
        d['resample'] = _pil_interpolation_to_str[d['resample']]
        return s.format(name=self.__class__.__name__, **d)


class SegmentationNormalize(SegmentationTransform):
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.normalize_transform = torchvision.transforms.Normalize(mean, std, inplace)

    def __call__(self, image: Image, target: Image):
        image = self.normalize_transform(image)
        return image, target

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(
            self.normalize_transform.mean, self.normalize_transform.std)


class SegmentationRandomNoise(SegmentationTransform):
    def __init__(self, noise_type='gauss', noise_params=None):
        super().__init__()
        if noise_params is None:
            noise_params = {}
        self.noise_type = noise_type
        self.noise_params = noise_params

    def _noisy(self, image):
        """
        Parameters
        ----------
        image : ndarray
            Input image data. Will be converted to float.
        mode : str
            One of the following strings, selecting the type of noise to add:

            'gauss'     Gaussian-distributed additive noise.
            'poisson'   Poisson-distributed noise generated from the data.
            's&p'       Replaces random pixels with 0 or 1.
            'speckle'   Multiplicative noise using out = image + n*image,where
                        n is uniform noise with specified mean & variance.
        """
        if self.noise_type == 'gauss':
            row, col, ch = image.shape
            mean = self.noise_params.get('mean', 0.0)
            var = self.noise_params.get('var', 1.0)
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = image + gauss
            return noisy
        elif self.noise_type == 's&p':
            row, col, ch = image.shape
            s_vs_p = self.noise_params.get('s_vs_p', 0.5)
            amount = self.noise_params.get('amount', 0.004)
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            out[coords] = 0
            return out
        elif self.noise_type == 'poisson':
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            return noisy
        elif self.noise_type == 'speckle':
            row, col, ch = image.shape
            gauss = np.random.randn(row, col, ch)
            gauss = gauss.reshape(row, col, ch)
            noisy = image + image * gauss
            return noisy

    def __call__(self, image: Image, target: Image):
        image = np.array(image, dtype=np.int16)
        image = self._noisy(image)
        image[image < 0.0] = 0.0
        image[image > 255.0] = 255.0
        image = image.round().astype(np.uint8)
        image = Image.fromarray(image)
        return image, target

    def __repr__(self):
        return self.__class__.__name__ + '(noise_type={0})'.format(self.noise_type)
