from Data.get_data import get_data
from utils.check_dataset import check_dataset



if __name__=='__main__':
    image, image_test, segmentation, segmentation_test = get_data()
    check_dataset()
    
