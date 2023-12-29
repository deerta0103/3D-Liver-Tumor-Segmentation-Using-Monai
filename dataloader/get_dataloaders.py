from monai.data import pad_list_data_collate
from monai.data import DataLoader
from transforms.add_transforms import get_dataset
from Data.get_data import get_data
from utils.check_dataset import check_dataset


def get_dataloaders(batch_size, num_classes):
    image_train, image_val, image_test, segmentation_train, segmentation_val, segmentation_test = get_data()
    print(f"Image number {len(image_train)}")
    check_dataset(image_train, segmentation_train)

    train_dataset = get_dataset(image_train, segmentation_train,num_classes=num_classes, flag="train")
    val_dataset = get_dataset(image_val, segmentation_val,num_classes=num_classes, flag="test")

# note that when batch size is only 1 then in the dataloader, collate_fn will not be able to make equal length of all the vectors as smaller vectors having shape 94,94,94 will be padded to highest among the padded vectors. 
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4,collate_fn=pad_list_data_collate, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4,collate_fn=pad_list_data_collate)

    return train_dataloader, val_dataloader, image_test, segmentation_test