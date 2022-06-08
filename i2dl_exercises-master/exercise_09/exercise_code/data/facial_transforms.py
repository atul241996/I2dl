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


class FacialTransform(ABC):
    def __init__(self):
        super(FacialTransform, self).__init__()

    @abstractmethod
    def __call__(self, image: Image, keypoints: torch.Tensor):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class FacialCompose(FacialTransform):
    def __init__(self, transforms):
        super(FacialCompose, self).__init__()
        self.transforms = transforms

    def __call__(self, image: Image, keypoints: torch.Tensor):
        for transform in self.transforms:
            image, keypoints = transform(image, keypoints)
        return image, keypoints

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class FacialResize(FacialTransform):
    def __init__(self, size, interpolation=Image.BILINEAR):
        super(FacialResize, self).__init__()
        self.resize_transform = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, image: Image, keypoints: torch.Tensor):
        image = self.resize_transform(image)
        return image, keypoints

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.resize_transform.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(
            self.resize_transform.size, interpolate_str)


class FacialToTensor(FacialTransform):
    def __init__(self):
        super(FacialToTensor, self).__init__()
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, image: Image, keypoints: torch.Tensor):
        image = self.to_tensor(image)
        return image, keypoints

    def __repr__(self):
        return self.__class__.__name__ + '()'


class FacialRandomHorizontalFlip(FacialTransform):
    def __init__(self, p=0.5):
        super(FacialRandomHorizontalFlip, self).__init__()
        self.p = p

    def __call__(self, image: Image, keypoints: torch.Tensor):
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            keypoints[:, 0] = -keypoints[:, 0]
        return image, keypoints

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class FacialColorJitter(FacialTransform):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super(FacialColorJitter, self).__init__()
        self.color_jitter_transform = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image: Image, keypoints: torch.Tensor):
        image = self.color_jitter_transform(image)
        return image, keypoints

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.color_jitter_transform.brightness)
        format_string += ', contrast={0}'.format(self.color_jitter_transform.contrast)
        format_string += ', saturation={0}'.format(self.color_jitter_transform.saturation)
        format_string += ', hue={0})'.format(self.color_jitter_transform.hue)
        return format_string


class FacialRandomAffine(FacialTransform):
    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0):
        super(FacialRandomAffine, self).__init__()
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
        matrix = FacialRandomAffine._get_inverse_affine_matrix(center, angle, translate, scale, shear)
        kwargs = {"fillcolor": fillcolor} if PILLOW_VERSION[0] >= '5' else {}
        return img.transform(output_size, Image.AFFINE, matrix, resample, **kwargs)

    def __call__(self, image: Image, keypoints: torch.Tensor):
        angle, translations, scale, shear = self.get_params(self.degrees, self.translate, self.scale, self.shear, image.size)
        image = FacialRandomAffine.affine(image, angle, translations, scale, shear, resample=self.resample, fillcolor=self.fillcolor)
        # affine transform the keypoints
        center = (0.0, 0.0)
        translations = tuple(2 * translation / image.size[i] for i, translation in enumerate(translations))
        m_inv = FacialRandomAffine._get_inverse_affine_matrix(center, angle, translations, scale, shear)
        m_inv = torch.tensor(m_inv + [0, 0, 1], dtype=torch.float).reshape(3, 3)
        m = m_inv.inverse()
        keypoints = torch.cat([keypoints, torch.ones(keypoints.shape[0], 1)], dim=1)
        keypoints = (m @ keypoints.T).T[:, :2]
        return image, keypoints

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


class FacialNormalize(FacialTransform):
    def __init__(self, mean, std, inplace=False):
        super(FacialNormalize, self).__init__()
        self.normalize_transform = torchvision.transforms.Normalize(mean, std, inplace)

    def __call__(self, image: Image, keypoints: torch.Tensor):
        image = self.normalize_transform(image)
        return image, keypoints

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(
            self.normalize_transform.mean, self.normalize_transform.std)
