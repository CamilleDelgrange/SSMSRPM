# This file is part of From Barlow Twins to Triplet Training: Differentiating Dementia with Limited Data (Triplet Training).
#
# Triplet Training is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Triplet Training is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Triplet Training. If not, see <https://www.gnu.org/licenses/>.

import monai.transforms as montrans
import torchio as tio
import random

class SSLTransform:
    def __init__(self, 
                 is_training=True, 
                 original_height=182, 
                 input_height=64, 
                 roi_size=None):
        
        self.original_height = original_height
        self.input_height = input_height
        self.is_training = is_training
        self.roi_size = roi_size

        self.base_transform = montrans.Compose([
            montrans.RandSpatialCrop(roi_size=(roi_size, roi_size, roi_size),
                                     random_center=True, random_size=False), #lazy=True
            montrans.Resize((self.input_height, self.input_height, self.input_height), mode = 'trilinear', align_corners=False), 
            montrans.ScaleIntensity(minv=0.0, maxv=1.0),
        ])

        self.augment_transform = montrans.Compose([
            #tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
            montrans.RandFlip(prob=0.5, spatial_axis=(0,1,2)),
            montrans.RandAffine(
                prob=0.5,
                translate_range=(8, 8, 8),  # +-8 pixels offset in each dimension
                rotate_range=(90, 90, 90),  # +-90 degrees rotation in each dimension
                scale_range=(0.95, 1.05),  # Scales between 0.95 and 1.05
                padding_mode='border',  # Use border padding
                mode=('trilinear'),  # Use bilinear interpolation for images, nearest for labels
            ),
            #tio.RandomAffine(
                #scales=0.05,
                #degrees=90,  # +-90 degrees in each dimension
                #translation=8,  # +-8 pixels offset in each dimension
                #image_interpolation="linear",
                #default_pad_value="otsu",
                #p=0.5,
            #),
        ])

    def __call__(self, sample):
        crop = self.base_transform(sample)
        
        # Decide randomly whether to apply augmentation or not
        if random.random() < 0.5:
            crop = self.augment_transform(crop)
        
        return crop