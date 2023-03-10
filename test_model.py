import os
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array, image_dataset_from_directory, load_img
import numpy as np
from utils import return_classes


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

train_model = load_model(
    r"E:\DICOM\medical dataset\New folder\model_weight\xceptionModel_50.h5")
# print(train_model.summary())


def return_test_ds(input_dir):
    test_ds = image_dataset_from_directory(
        input_dir, seed=42, image_size=(224, 224), batch_size=16, label_mode='categorical')
    return test_ds


def predict_img(img_path):
    classes = return_classes(r"E:\DICOM\medical dataset\annotation.txt")
    model = train_model
    img = load_img(img_path, target_size=(224, 224))
    # img = np.array(img)
    disp_img = cv2.imread(img_path)
    img = img_to_array(img)
    # print(f'img_array{img}')
    img = np.expand_dims(img, axis=0)
    print(f"expand_dim{img.shape}")
    prediction = model.predict(img)
    probabilty = np.max(prediction)
    predict_class_id = np.argmax(prediction)
    predict_class = classes[predict_class_id]

    return predict_class, probabilty, disp_img


def evaluate_model():
    test_ds = return_test_ds(r"E:\DICOM\medical dataset\archive\test")
    model = train_model
    scr = model.evaluate(test_ds, verbose=0)

    return scr[0], scr[1],scr # loss and accuracy


if __name__ == "__main__":
    img_path = r"E:\DICOM\medical dataset\archive\test\normal-pylorus\3e9878ff-0764-47ba-9e20-bcfff33dc69e.jpg"
    predict_class, predict_probability, disp_img = predict_img(img_path)
    print(
        f"predict_class,predict_probability: {predict_class,predict_probability}")
    loss, acc,val = evaluate_model()
    print(f"loss and the accuracy: {loss,acc,val}")
    plt.imshow(cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB))
    plt.title(predict_class)
    plt.show()

