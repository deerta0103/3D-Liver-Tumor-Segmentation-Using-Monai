from monai.transforms import (AsDiscreted,RandShiftIntensityd,RandScaleIntensityd,ScaleIntensity,NormalizeIntensityd,Compose,ScaleIntensityd, Spacingd, Transform, EnsureChannelFirstd,AsChannelLast,LoadImaged, RandSpatialCropd, RandGaussianNoised,SpatialPadd,RandSpatialCropd,RandFlipd,EnsureTyped)
from monai.data import Dataset

class DatasetClass(Dataset):
    def __init__(self, file_names, transform = None):
        self.file_names = file_names 
        self.transform = transform
        
    def __getitem__(self,index):
        file_name = self.file_names[index]
        dataset = self.transform(file_name) 
        return dataset
            
    def __len__(self):
        return len(self.file_names)

def get_dataset(image, segmentation,num_classes, flag = "train"):
    
    if flag == "train":
        train_transforms = Compose([
            LoadImaged(keys=['image', 'seg'],image_only=False),
            EnsureChannelFirstd(keys=['image', 'seg']),
            Spacingd(keys=['image', 'seg'], pixdim=(1.0,1.0,1.0), mode=('bilinear','nearest')),
            RandFlipd(keys=['image','seg'], prob=0.5, spatial_axis=0),
            RandFlipd(keys=['image','seg'], prob=0.5, spatial_axis=1),
            RandFlipd(keys=['image','seg'], prob=0.5, spatial_axis=2),
            RandGaussianNoised(keys=['image']),
            RandSpatialCropd(['image', 'seg'],roi_size=(128, 128, 128), random_size=False),
            ScaleIntensityd(keys='image',channel_wise=False),
            AsDiscreted(keys='seg',to_onehot = num_classes),
            SpatialPadd(keys = ['image', 'seg'], spatial_size=(128,128,128))])

        file_names_train = []
        for i in range(len(image)):
            file_names_train.append({"image":image[i], "seg":segmentation[i]})
        train_ds = DatasetClass(file_names_train,train_transforms)
        return train_ds
    else:
        val_transforms = Compose([
            LoadImaged(keys=['image', 'seg'],image_only=False),
            EnsureChannelFirstd(keys=['image', 'seg']),
            Spacingd(keys=['image', 'seg'], pixdim=(1.0,1.0,1.0), mode=('bilinear','nearest')),
            RandSpatialCropd(['image', 'seg'],roi_size=(128, 128, 128), random_size=False),
            ScaleIntensityd(keys='image',channel_wise=False),
            AsDiscreted(keys='seg',to_onehot = num_classes),
            SpatialPadd(keys = ['image', 'seg'], spatial_size=(128,128,128))])
        
        file_names_val = []
        for i in range(len(image)):
            file_names_val.append({"image":image[i], "seg":segmentation[i]})
        val_ds = DatasetClass(file_names_val,val_transforms)
        return val_ds


