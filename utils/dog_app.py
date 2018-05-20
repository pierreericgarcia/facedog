import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Sequential
from keras.models import model_from_json
from PIL import Image
from keras import backend as K
import tensorflow as tf

dog_names = [
    'Affenpinscher', 'Afghan_hound', 'Airedale_terrier', 'Akita',
    'Alaskan_malamute', 'American_eskimo_dog', 'American_foxhound',
    'American_staffordshire_terrier', 'American_water_spaniel',
    'Anatolian_shepherd_dog', 'Australian_cattle_dog', 'Australian_shepherd',
    'Australian_terrier', 'Basenji', 'Basset_hound', 'Beagle',
    'Bearded_collie', 'Beauceron', 'Bedlington_terrier', 'Belgian_malinois',
    'Belgian_sheepdog', 'Belgian_tervuren', 'Bernese_mountain_dog',
    'Bichon_frise', 'Black_and_tan_coonhound', 'Black_russian_terrier',
    'Bloodhound', 'Bluetick_coonhound', 'Border_collie', 'Border_terrier',
    'Borzoi', 'Boston_terrier', 'Bouvier_des_flandres', 'Boxer',
    'Boykin_spaniel', 'Briard', 'Brittany', 'Brussels_griffon', 'Bull_terrier',
    'Bulldog', 'Bullmastiff', 'Cairn_terrier', 'Canaan_dog', 'Cane_corso',
    'Cardigan_welsh_corgi', 'Cavalier_king_charles_spaniel',
    'Chesapeake_bay_retriever', 'Chihuahua', 'Chinese_crested',
    'Chinese_shar-pei', 'Chow_chow', 'Clumber_spaniel', 'Cocker_spaniel',
    'Collie', 'Curly-coated_retriever', 'Dachshund', 'Dalmatian',
    'Dandie_dinmont_terrier', 'Doberman_pinscher', 'Dogue_de_bordeaux',
    'English_cocker_spaniel', 'English_setter', 'English_springer_spaniel',
    'English_toy_spaniel', 'Entlebucher_mountain_dog', 'Field_spaniel',
    'Finnish_spitz', 'Flat-coated_retriever', 'French_bulldog',
    'German_pinscher', 'German_shepherd_dog', 'German_shorthaired_pointer',
    'German_wirehaired_pointer', 'Giant_schnauzer', 'Glen_of_imaal_terrier',
    'Golden_retriever', 'Gordon_setter', 'Great_dane', 'Great_pyrenees',
    'Greater_swiss_mountain_dog', 'Greyhound', 'Havanese', 'Ibizan_hound',
    'Icelandic_sheepdog', 'Irish_red_and_white_setter', 'Irish_setter',
    'Irish_terrier', 'Irish_water_spaniel', 'Irish_wolfhound',
    'Italian_greyhound', 'Japanese_chin', 'Keeshond', 'Kerry_blue_terrier',
    'Komondor', 'Kuvasz', 'Labrador_retriever', 'Lakeland_terrier',
    'Leonberger', 'Lhasa_apso', 'Lowchen', 'Maltese', 'Manchester_terrier',
    'Mastiff', 'Miniature_schnauzer', 'Neapolitan_mastiff', 'Newfoundland',
    'Norfolk_terrier', 'Norwegian_buhund', 'Norwegian_elkhound',
    'Norwegian_lundehund', 'Norwich_terrier',
    'Nova_scotia_duck_tolling_retriever', 'Old_english_sheepdog', 'Otterhound',
    'Papillon', 'Parson_russell_terrier', 'Pekingese', 'Pembroke_welsh_corgi',
    'Petit_basset_griffon_vendeen', 'Pharaoh_hound', 'Plott', 'Pointer',
    'Pomeranian', 'Poodle', 'Portuguese_water_dog', 'Saint_bernard',
    'Silky_terrier', 'Smooth_fox_terrier', 'Tibetan_mastiff',
    'Welsh_springer_spaniel', 'Wirehaired_pointing_griffon', 'Xoloitzcuintli',
    'Yorkshire_terrier'
]

graph = tf.get_default_graph()

with graph.as_default():
    json_file = open('saved_models/first_layers.json', 'r')
    first_layers_model_json = json_file.read()
    json_file.close()
    first_layers_model = model_from_json(first_layers_model_json)
    first_layers_model.load_weights("saved_models/first_layers.h5")

    last_layers = Sequential()
    last_layers.add(GlobalAveragePooling2D(input_shape=(1, 1, 2048)))
    last_layers.add(Dense(133, activation='softmax'))

    last_layers.load_weights('saved_models/last_layers.hdf5')

    last_layers.compile(
        optimizer="rmsprop",
        loss='categorical_crossentropy',
        metrics=['accuracy'])


def path_to_tensor(img):
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    resized_img = img.resize((224, 224), Image.ANTIALIAS)
    x = image.img_to_array(resized_img, data_format=None)
    # convert 3D to 4D tensor with shape (1, 224, 224, 3) return 4D tensor
    output = np.expand_dims(x, axis=0)
    return output


def breed_detector(img):
    with graph.as_default():
        # extract bottleneck features
        bottleneck_feature = first_layers_model.predict(
            preprocess_input(path_to_tensor(img)))
        # obtain predicted vector
        predicted_vector = last_layers.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
        return dog_names[np.argmax(predicted_vector)]


def dog_app(img):
    return {'breed': breed_detector(img)}
