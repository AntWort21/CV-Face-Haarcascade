import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def get_train_image(path):
    '''
        To get a list of train images, images label, and images index using the given path

        Parameters
        ----------
        path : str
            Location of train root directory
        
        Returns
        -------
        list
            List containing all the train images
        list
            List containing all train images label
        list
            List containing all train images indexes
    '''
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
    '''
        To get a list of test subdirectories using the given path

        Parameters
        ----------
        path : str
            Location of test root directory
        
        Returns
        -------
        list
            List containing all the test subdirectories
    '''
    directories = []

    for i in os.listdir(path):
        directories.append(i)

    return directories


def get_all_test_images(path):
    '''
        To load a list of test images from given path list. Resize image height 
        to 200 pixels and image width to the corresponding ratio for train images

        Parameters
        ----------
        path : str
            Location of images root directory
        
        Returns
        -------
        list
            List containing all image that has been resized for each Test Folders
    '''
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
    '''
        To detect a face from given image list and filter it if the face on
        the given image is not equals to one

        Parameters
        ----------
        faces_list : list
            List containing all loaded images
        labels_list : list
            List containing all image classes labels
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            list containing image gray face location
        list
            List containing all filtered image classes label
    '''
    face_list_gray = []
    label_list_gray = []
    image_gray_location = []

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
    '''
        To create and train face recognizer object

        Parameters
        ----------
        grayed_images_list : list
            List containing all filtered and cropped face images in grayscale
        grayed_labels : list
            List containing all filtered image classes label
        
        Returns
        -------
        object
            Recognizer object after being trained with cropped face images
    '''
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
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
    '''
        To check the predicted user is in the designed room or not

        Parameters
        ----------
        predicted_name : str
            The name result from predicted user
        room_number : int
            The room number that the predicted user entered

        Returns
        -------
        bool
            If predicted user entered the correct room return True otherwise False
    '''
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
    '''
        To draw prediction and validation results on the given test images

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories
        room: int
            The room number

        Returns
        -------
        list
            List containing all test images after being drawn with
            its prediction and validation results
    '''
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
    '''
        To show the final image that already combine into one image

        Parameters
        ----------
        room : str
            The room number in string format (e.g. 'Room 1')
        predicted_test_image_list : nparray
            Array containing image data
    '''
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