import os
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T
import pandas as pd
from PIL import Image
np.seterr(invalid='ignore')

def extract_mip(image, d=10, s=40):
    image_c = image.copy()

    image_c[:, :s, ] = 0
    image_c[:, -s:, ] = 0
    image_c[:, :, :s] = 0
    image_c[:, :, -s:] = 0

    (_, _, Z) = np.meshgrid(range(image.shape[1]), range(image.shape[0]), range(image.shape[2]))
    M = Z * (image_c > 0)
    M = M.sum(axis=2) / (image_c > 0).sum(axis=2)
    M[np.isnan(M)] = 0
    mask = M > 0
    c = int(np.mean(M[mask]))

    image_frontal = np.max(image_c, axis=1)
    image_sagittal = np.max(image_c[:, :, c - d:c + d], axis=2)[::-1, :]

    return image_frontal, image_sagittal



def extract_mode_slice(sitk_image, target_spacing=1, mode='frontal', min_height=512,
                                              min_width=512):
    spacing = sitk_image.GetSpacing()
    direction = sitk_image.GetDirection()
    dx = int(direction[0])
    dy = int(direction[4])
    dz = int(direction[8])

    image = sitk.GetArrayFromImage(sitk_image)[::dx, ::dy, ::dz]
    image = np.int16(image)
    image_frontal, image_sagittal = extract_mip(image)

    if mode == 'sagittal':
        image = image_sagittal
    else:
        image = image_frontal

    return image


def resize_image(itk_image, newSize, resamplemethod = sitk.sitkLinear):
    resampler = sitk.ResampleImageFilter()
    originSize = itk_image.GetSize()
    originSpacing = itk_image.GetSpacing()

    newSize = np.array(newSize, float)
    scale_factor = originSize / newSize
    newSpacing = originSpacing * scale_factor
    newSize = np.array(newSize).astype(np.uint16)

    resampler.SetReferenceImage(itk_image)
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itk_img_resize = resampler.Execute(itk_image)

    return itk_img_resize, newSpacing

def ct_normalize(arr_image, window_level=50, window_width=350, epsilon=1e-6):
    min_v = window_level - window_width // 2
    max_v = window_level + window_width // 2
    arr_image = (arr_image.clip(min_v, max_v) - min_v) / (max_v - min_v + epsilon)
    aug_arr_image = (arr_image * 255).astype(np.uint8)

    return aug_arr_image

 
class LocL3Dataset(Dataset):
    def __init__(self, args, split, printer=print):
        super().__init__()
        self.data_dir = args.data_dir
        self.split = split
        self.target_size = args.target_size # (D, H, W)
        self.split_ratio = args.split_ratio
        self.sigma = args.sigma
        self.printer = printer
        if self.split in ['train',]:
            pd_data = pd.read_csv(os.path.join(self.data_dir, 'loc_info_tr.csv'))
        else:
            pd_data = pd.read_csv(os.path.join(self.data_dir, 'loc_info_val.csv'))
        self.samples = pd_data.values

        if args.strong_aug:
            self.transform = T.Compose([T.RandomRotation(30),
                                        T.RandomVerticalFlip(p=0.5),
                                        T.RandomHorizontalFlip(p=0.5),
                                        T.RandomAffine(0, (0.1, 0.1)),
                                        T.ToTensor(),])
        else:
            self.transform = T.Compose([T.ToTensor(),])
        
        self._summary()

    def generate_target(self, z_start, z_end):
        target_start = np.zeros((1, 
                                int(self.target_size[0]//2 * self.split_ratio)), 
                                dtype=np.float32)
        target_end = np.zeros((1,
                              int(self.target_size[0]//2 * self.split_ratio)),
                              dtype=np.float32)
        
        mu_s, mu_e = z_start * self.split_ratio, z_end * self.split_ratio
        assert z_start <= 256
        assert z_end >= 0
        s = np.arange(0, 256, 1, np.float32)
        e = np.arange(0, 256, 1, np.float32)
        
        target_start[0] = (np.exp(-((s - mu_s) ** 2) / (2 * self.sigma ** 2))) / (self.sigma * np.sqrt(np.pi*2))
        target_end[0] = (np.exp(-((e - mu_e) ** 2) / (2 * self.sigma ** 2))) / (self.sigma * np.sqrt(np.pi*2))
        #target_start[0,int(mu_s)] = 1
        #target_end[0,int(mu_e)] = 1
        return target_start, target_end
    
    def _summary(self):
        self.printer(f"[{self.split}]\tLoaded {self.__len__()} samples")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_id, sample_time, z_start, z_end = self.samples[idx]
        itk_image = sitk.ReadImage(os.path.join(self.data_dir, str(sample_id)+'.nii.gz'))
        z = float(itk_image.GetSize()[-1])
        z_new_start = round(float(z_start / z) * self.target_size[0] * self.split_ratio)
        z_new_end = round(float(z_end / z) * self.target_size[0] * self.split_ratio) - 256
        target_start, target_end = self.generate_target(z_new_start, z_new_end)
        spacing = itk_image.GetSpacing()
        resize_itk_image, newSpacing = resize_image(itk_image, (64, 64, 512))
        start = float(z_new_start * newSpacing[-1])
        end = float(z_new_end * newSpacing[-1])
        arr_image = sitk.GetArrayFromImage(resize_itk_image)
        # [512, 64, 64]
        image = ct_normalize(arr_image)
        

        front_image = ct_normalize(arr_image[:,32,:]) 
        sagit_image = ct_normalize(arr_image[:,:,32]) 
        
        front_image_3c = np.stack([front_image, front_image, front_image],axis=-1)
        sagit_image_3c = np.stack([sagit_image, sagit_image, sagit_image],axis=-1)
        front_image_3c = Image.fromarray(front_image_3c)
        front_image = self.transform(front_image_3c)
        sagit_image_3c = Image.fromarray(sagit_image_3c)
        sagit_image = self.transform(sagit_image_3c)

        images = []
        for i in range(image.shape[0]):
            aug_slice_3c = np.stack([image[i,...], image[i,...], image[i,...]], axis=-1)
            aug_slice_3c = Image.fromarray(aug_slice_3c)
            aug_slice_3c = self.transform(aug_slice_3c)
            images.append(aug_slice_3c[0:1,...])
        image = torch.stack(images, dim=1)
        
        return {
            "image": [image, front_image[0:1,...], sagit_image[0:1,...]],
            "target_start": target_start,
            "target_end": target_end,
            "start": start,
            "end": end,
            "z_start":z_new_start,
            "z_end": z_new_end,
            "spacing": newSpacing,
            "id":sample_id
        }
        



