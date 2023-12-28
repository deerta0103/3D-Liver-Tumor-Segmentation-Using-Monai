from monai.data import pad_list_data_collate
from monai.data import Dataset, DataLoader
from transforms.train_transforms import get_train_dataset
from torch.utils.data import random_split
from Data.get_data import get_data
from utils.check_dataset import check_dataset




def split_train_and_val_data(dataset):
    total_samples = len(dataset)
    # print(total_samples)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    return train_ds, val_ds

def get_dataloaders(batch_size, num_classes):
    image, image_test, segmentation, segmentation_test = get_data()
    print(f"Image number {len(image)}")
    check_dataset(image, segmentation)
    dataset = get_train_dataset(image, segmentation, num_classes=num_classes)
    train_ds, val_ds = split_train_and_val_data(dataset)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=4,collate_fn=pad_list_data_collate, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, num_workers=4,collate_fn=pad_list_data_collate, shuffle=True)

    return train_dataloader, val_dataloader, image_test, segmentation_test