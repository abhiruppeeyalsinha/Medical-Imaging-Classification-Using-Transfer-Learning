import os,matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import keras
from keras.layers import Conv2D, BatchNormalization, Dense, Activation, GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, Conv2DTranspose
from keras.preprocessing.image import ImageDataGenerator, image_dataset_from_directory
from keras.models import Model
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import SGD 


from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet121
from keras.applications.inception_v3 import InceptionV3
from utils import return_classes
# from config import config


input_dim = 224

# data_img = r"E:\DICOM\medical dataset\archive\train"
# for i in os.listdir(data_img):
#     print(i)
#     # for j in  os.listdir(data_img +'/'+ i):
#         # print(f"j:- {j}")
#         # img_file = cv2.imread(os.path.join(data_img+'/'+i, j))
#         # cv2.imshow(str(j), img_file)
#         # cv2.waitKey(1000)
#         # cv2.destroyAllWindows()
#     pass


def return_ds(input_dir):
    train_dataset = image_dataset_from_directory(
        input_dir, validation_split=.2, subset="training", seed=42,
        image_size=(input_dim, input_dim), batch_size=16, label_mode='categorical')

    val_dataset = image_dataset_from_directory(
        input_dir, validation_split=.2, subset="validation", seed=42,
        image_size=(input_dim, input_dim), batch_size=8, label_mode='categorical')

    return train_dataset, val_dataset


def return_model(input_dim, nb_classes, freeze=False, head=None):
    if head == 'xception' or head == 'Xception':
        base_model = Xception(
            include_top=False, pooling=GlobalMaxPooling2D, weights='imagenet', input_shape=(input_dim, input_dim, 3))
    elif head == 'inceptionv3' or head == 'Inceptionv3':
        base_model = InceptionV3(
            include_top=False, weights='imagenet', input_shape=(input_dim, input_dim, 3))
    elif head == 'vgg166' or head == 'VGG16':
        base_model = VGG16(
            include_top=False, weights='imagenet', input_shape=(input_dim, input_dim, 3))
    elif head == 'DenseNet121' or head == 'denseNet121':
        base_model = DenseNet121(
            include_top=False, weights='imagenet', input_shape=(input_dim, input_dim, 3))
    if head == None:
        print("Please choose the pretained model")

    if freeze:
        for layer in base_model.layers:
            # print('layer:-',layer)
            layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
        # x = Dense(1024, activation='sigmoid')(x)

    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(inputs =base_model.inputs, outputs=predictions)

    return model


# return_model((224,224,3),1000,False,'xception')
if __name__ == '__main__':
    classes = return_classes(r"E:\DICOM\medical dataset\annotation.txt")

    train_ds, val_ds = return_ds(r"E:\DICOM\medical dataset\archive\train")
    model = return_model(224, len(classes), head='xception', freeze=False
                         )
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])
    save_weight = ModelCheckpoint(
        r"E:\DICOM\medical dataset\model_weight\model_weight\eptionModel_50.h5",monitor='val_accuracy',
        verbose=1, save_best_only=True,save_weights_only=False,mode='max')


    history = model.fit(train_ds,epochs=50,validation_data = val_ds,callbacks=[save_weight])






#     plt.plot(history.history['accuracy'])
#     plt.plot(history.history['val_accuracy'])
#     plt.title("Model Accuracy")
#     plt.ylabel("Accuracy")
#     plt.xlabel("Epoch")
#     plt.legend(['Train','Test'],loc='uppper left')
#     plt.show()
