from monai.transforms import (AsDiscreted,RandShiftIntensityd,RandScaleIntensityd,ScaleIntensity,NormalizeIntensityd,Compose,ScaleIntensityd, Spacingd, Transform, EnsureChannelFirstd,AsChannelLast,LoadImaged, RandSpatialCropd, RandGaussianNoised,SpatialPadd,RandSpatialCropd,RandFlipd,EnsureTyped)


def get_train_dataset(image, segmentation,num_classes):
    
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
    AsDiscreted(keys='seg',to_onehot = num_classes)])

    file_names = []

    for i in range(len(image)):
        file_names.append({"image":image[i], "seg":segmentation[i]})

    dataset = train_transforms(file_names)
    return dataset


