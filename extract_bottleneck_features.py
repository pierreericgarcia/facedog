def extract_Resnet50(tensor):
    from keras.applications.resnet50 import ResNet50, preprocess_input
    return ResNet50(
        weights='imagenet', include_top=False).predict(
            preprocess_input(tensor))