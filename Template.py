import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def get_train_image(path):

    #path = 'Dataset/Train'

    face_list = []
    face_list_id = []
    face_dir_list = os.listdir(path)

    for idx, train_dir in enumerate(face_dir_list):
        face_path_list = os.listdir(path + '/' + train_dir)
        for face_path in face_path_list:#get into each individual folder
            image = cv2.imread(path + '/' + train_dir + "/"+ face_path)
            face_list.append(image) #get each person image
            face_list_id.append(idx) #index of the person

    return face_list, face_dir_list, face_list_id


def get_all_test_folders(path):
    #path = 'Dataset/Test'
    directories = []

    for i in os.listdir(path):
        directories.append(i)

    return directories


def get_all_test_images(path):


    #path = 'Dataset/Test/Room 1'
    resized_img_list = []

    for img_name in os.listdir(path):#get into individual rooms
        img = cv2.imread(path + "/" + img_name)
        ratio = 200/img.shape[0]
        dimension = (int(img.shape[1]*ratio), 200)
        resized_img = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
        resized_img_list.append(resized_img)

    return resized_img_list


def detect_faces_and_filter(faces_list, labels_list=None):
    #grayed_trained_images_list, _, grayed_trained_labels_list = detect_faces_and_filter(faces_list, indexes_list)
    # return face_list, face_list_id
    # face_list -> each person image
    #face list_id -> index of the person

    face_list_gray = []
    label_list_gray = []
    image_gray_location = []

    # print(labels_list)

    if(labels_list != None): #Training
        for (idx, img) in zip(labels_list, faces_list):
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected_face = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)
            if(len(detected_face) < 1):
                continue
            for face_rect in detected_face:
                x,y,h,w = face_rect
                face_img = img_gray[y:y+h, x:x+w]
                image_gray_location.append(face_rect)
                face_list_gray.append(face_img)
                label_list_gray.append(idx)

    else: #Testing 
        for img in faces_list:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected_face = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)
            if(len(detected_face) < 1):
                continue
            for face_rect in detected_face:
                x,y,h,w = face_rect
                face_img = img_gray[y:y+h, x:x+w]
                image_gray_location.append(face_rect)
                face_list_gray.append(face_img)
            
    return face_list_gray, image_gray_location, label_list_gray  

def train(grayed_images_list, grayed_labels_list):

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    # face_recognizer = cv2.face.EigenFaceRecognizer_create()
    # face_recognizer = cv2.face.FisherFaceRecognizer_create()

    face_recognizer.train(grayed_images_list, np.array(grayed_labels_list))

    return face_recognizer


def predict(recognizer, gray_test_image_list):
    '''
        To predict the test image with the recognizer

        Parameters
        ----------
        recognizer : object
            Recognizer object after being trained with cropped face images
        gray_test_image_list : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''

    result_confidence_list = []

    for img in gray_test_image_list:
        result, confidence = recognizer.predict(img)
        result_confidence_list.append([result, confidence])

    return result_confidence_list

    


def check_attandee(predicted_name, room_number):
    room1participant = [
        'Elon Musk',
        'Steve Jobs',
        'Benedict Cumberbatch',
        'Donald Trump'
    ]

    room2participant = [
        'IU',
        'Kim Se Jeong',
        'Kim Seon Ho',
        'Rich Brian'
    ]

    if(room_number == 1 and predicted_name not in room1participant):
        return False
    
    if(room_number == 2 and predicted_name not in room2participant):
        return False
    
    return True

    
def write_prediction(predict_results, test_image_list, test_faces_rects, train_names, room):
    predicted_test_image_list = []

    for image, face_rect, [res, conf] in zip(test_image_list, test_faces_rects, predict_results):
        x,y,h,w = face_rect
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.rectangle(image, (x,y), (x+w, y+h), (255,255,0), 1)
        if(check_attandee(train_names[res], room)):
            text = train_names[res] + ' : ' + str("Present")
            cv2.putText(image, text, (x-65,y+5), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)
        else:
            text = train_names[res] + ' : ' + str("Shouldn't be here")
            cv2.putText(image, text, (x-65,y+5), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
        predicted_test_image_list.append(image)
    
    return predicted_test_image_list



def combine_and_show_result(room, predicted_test_image_list):
    image_merge = []
    for image in predicted_test_image_list:
        ratio = 200/image.shape[0]
        dimension = (int(image.shape[1]*ratio), 200)
        image = cv2.resize(image, dimension, interpolation=cv2.INTER_AREA)
        image_merge.append(image)

    final_image = np.concatenate(image_merge, axis=1)
    plt.figure(figsize=(12,6))
    plt.gcf().canvas.manager.set_window_title(room)
    plt.imshow(final_image)
    plt.show()




'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''

def main():
    
    '''
        Please modify train_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_path = 'Dataset/Train'
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    faces_list, labels_list, indexes_list = get_train_image(train_path)
    grayed_trained_images_list, _, grayed_trained_labels_list = detect_faces_and_filter(faces_list, indexes_list)
    recognizer = train(grayed_trained_images_list, grayed_trained_labels_list)

    '''
        Please modify test_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_path = 'Dataset/Test'
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    test_images_folder = get_all_test_folders(test_path)
    for index, room in enumerate(test_images_folder):
        test_images_list = get_all_test_images(test_path + '/' + room)
        grayed_test_image_list, grayed_test_location, _ = detect_faces_and_filter(test_images_list)
        predict_results = predict(recognizer, grayed_test_image_list)
        predicted_test_image_list = write_prediction(predict_results, test_images_list, grayed_test_location, labels_list, index+1)
        combine_and_show_result(room, predicted_test_image_list)


if __name__ == "__main__":
    main()