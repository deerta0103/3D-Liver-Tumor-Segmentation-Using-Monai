import glob 

def get_data():
    
    image = sorted(glob.glob('/media/shirshak/E076749B767473DE/LiverTumorSegmentationDataset/Task03_Liver_rs/imagesTr/*.nii'))
    segmentation = sorted(glob.glob('/media/shirshak/E076749B767473DE/LiverTumorSegmentationDataset/Task03_Liver_rs/labelsTr/*.nii'))

    image_train, image_val, image_test = image[:83], image[83:113], image[113:123]
    segmentation_train, segmentation_val, segmentation_test = segmentation[:83], segmentation[83:113], segmentation[113:123]

    return image_train, image_val, image_test, segmentation_train, segmentation_val, segmentation_test



# if __name__=="__main__":
#     image_train, image_val, image_test, segmentation_train, segmentation_val, segmentation_test = get_data()
#     print(len(image_train))








