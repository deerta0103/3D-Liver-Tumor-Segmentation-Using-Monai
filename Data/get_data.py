import glob 

def get_data():
    
    image = sorted(glob.glob('/media/shirshak/E076749B767473DE/LiverTumorSegmentationDataset/Task03_Liver_rs/imagesTr/*.nii'))
    segmentation = sorted(glob.glob('/media/shirshak/E076749B767473DE/LiverTumorSegmentationDataset/Task03_Liver_rs/labelsTr/*.nii'))

    image, image_test = image[:113], image[113:123]
    segmentation, segmentation_test = segmentation[:113], segmentation[113:123]

    return image, image_test, segmentation, segmentation_test












