# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import cv2
import numpy as np

from ...types import BatchType


def adjust_hsv(
    img_rgb: np.ndarray,
    h_offset: int,
    s_offset: int,
    v_offset: int
) -> np.ndarray:
    # we use OpenCV for converting to HSV space, thus, hue is in [0, 179]
    # (step size is 2 degrees), saturation is in [0, 255], and value is in
    # [0, 255].
    assert -180 <= h_offset <= 180
    assert -255 <= s_offset <= 255
    assert -255 <= v_offset <= 255

    # convert to HSV
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_h = img_hsv[:, :, 0]
    img_s = img_hsv[:, :, 1]
    img_v = img_hsv[:, :, 2]

    # apply adjustment
    img_h = ((img_h.astype('int') + h_offset) % 180).astype('uint8')
    img_s = np.clip(img_s.astype('int') + s_offset, 0, 255).astype('uint8')
    img_v = np.clip(img_v.astype('int') + v_offset, 0, 255).astype('uint8')
    img_hsv = np.stack([img_h, img_s, img_v], axis=2)

    # convert back to rgb
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)


class RandomHSVJitter:
    def __init__(
        self,
        hue_jitter: float,
        saturation_jitter: float,
        value_jitter: float
    ) -> None:
        """
        Random color jitter augmentation in HSV space

        Parameters
        ----------
        hue_jitter: float
            How much jitter hue (angle). The applied hue offset is chosen
            uniformly from [-hue_jitter, hue_jitter] with 0 <= hue_jitter
            <= 0.5. Note that 0 means original color (no hue jittering) and
            that both 0.5 and -0.5 result in the complementary colors due to
            the angle periodicity.
        saturation_jitter: float
            How much jitter saturation. The applied saturation offset is chosen
            uniformly from [-saturation_jitter, saturation_jitter] with
            0 <= saturation_jitter <= 1.
        value_jitter: float
            How much jitter value. The applied value offset is chosen uniformly
            from [-value_jitter, value_jitter] with 0 <= value_jitter <= 1.

        Notes
        -----
        This augmentation was fixed in EMSANet (and is different to ESANet)
        Processed images need to be of type uint8.
        """
        # we use OpenCV for converting to HSV space, thus, hue is in [0, 179]
        # (step size is 2 degrees), saturation is in [0, 255], and value is in
        # [0, 255].
        self._hue_limits = [int(-hue_jitter*(360/2)),
                            int(hue_jitter*(360/2))]
        self._saturation_limits = [int(-saturation_jitter*255),
                                   int(saturation_jitter*255)]
        self._value_limits = [int(-value_jitter*255),
                              int(value_jitter*255)]

    def __call__(self, sample: BatchType) -> BatchType:
        # this augmentation is applied to rgb image only
        if 'rgb' not in sample:
            return sample

        img = sample['rgb']
        assert img.dtype == 'uint8'

        # get offsets
        h_offset = np.random.randint(self._hue_limits[0],
                                     self._hue_limits[1])
        s_offset = np.random.randint(self._saturation_limits[0],
                                     self._saturation_limits[1])
        v_offset = np.random.randint(self._value_limits[0],
                                     self._value_limits[1])

        # apply augmentation
        sample['rgb'] = adjust_hsv(img, h_offset, s_offset, v_offset)

        return sample
