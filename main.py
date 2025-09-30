import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math

def image_processing():
    img = cv2.imread('bichon.jpg')
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', gray_image)
    cv2.waitKey(0)
    
    print("Please wait...")

    _, bin_thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
    _, inv_bin_thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)
    _, trunc_thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_TRUNC)
    _, tozero_thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_TOZERO)
    _, inv_tozero_thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_TOZERO_INV)
    _, otsu_thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_OTSU)

    result_image = [gray_image, bin_thresh, inv_bin_thresh, trunc_thresh, tozero_thresh, inv_tozero_thresh, otsu_thresh]
    result_desc = ['Grayscale', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV', 'OTSU']

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mean_blur = cv2.blur(img, (11, 11))
    gaussian_blur = cv2.GaussianBlur(img, (11, 11), 5.0)
    median_blur = cv2.medianBlur(img, 11)
    bilateral_blur = cv2.bilateralFilter(img, 5, 150, 150)

    result_image = [img, mean_blur, gaussian_blur, median_blur, bilateral_blur]
    result_desc = ['Original', 'Mean', 'Gaussian', 'Median', 'Bilateral']
    def manual_averaging(img, ksize):
        sub = ksize - 1
        np_img = np.array(img)
        for i in range(np_img.shape[0]-sub):
            for j in range(np_img.shape[1]-sub):
                arr = np.array(np_img[i:(i+ksize), j:(j+ksize)]).flatten()
                mean = np.mean(arr)
                np_img[i+ksize//2, j+ksize//2] = mean
        return np_img
    manual_averaging = manual_averaging(gray_image, 5)

    def manual_median(img, ksize):
        sub = ksize - 1
        np_img = np.array(img)
        for i in range(np_img.shape[0]-sub):
            for j in range(np_img.shape[1]-sub):
                arr = np.array(np_img[i:(i+ksize), j:(j+ksize)]).flatten()
                med = np.median(arr)
                np_img[i+ksize//2, j+ksize//2] = med
        return np_img
    manual_median = manual_median(gray_image, 5)

    result_image = [gray_image, manual_averaging, manual_median]
    result_desc = ['Grayscale', 'Manual Averaging', 'Manual Median']
    plt.figure(1, figsize=(8, 8))
    for i, (curr_image, curr_desc) in enumerate(zip(result_image, result_desc)):
        plt.subplot(1, 3, (i+1))
        plt.imshow(curr_image, 'gray')
        plt.title(curr_desc)
        plt.xticks([])
        plt.yticks([])
    plt.show()

def edge_detection():
    img = cv2.imread('bichon.jpg')
    img_h = img.shape[0]
    img_w = img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F) # float64
    laplacian_uint8 = np.uint8(np.absolute(laplacian))

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, 5)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, 5)

    canny = cv2.Canny(gray, 100, 200)
    
    result_image = [gray, laplacian, laplacian_uint8, sobel_x, sobel_y, canny]
    result_desc = ['Original', 'Laplacian', 'uint8', 'Sobel_X', 'Sobel_Y', 'Canny']
    plt.figure(1, figsize=(8, 8))
    for i, (curr_image, curr_desc) in enumerate(zip(result_image, result_desc)):
        plt.subplot(2, 3, (i+1))
        plt.imshow(curr_image, 'gray')
        plt.title(curr_desc)
        plt.xticks([])
        plt.yticks([])
    plt.show()
    
def shape_detection():
    img_object = cv2.imread('guitar.jpg')
    img_scene = cv2.imread('guitardeco.jpg')

    # Use ORB instead of SURF
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors with ORB
    kp_object, des_object = orb.detectAndCompute(img_object, None)
    kp_scene, des_scene = orb.detectAndCompute(img_scene, None)

    # FLANN parameters for ORB
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_object, des_scene, k=2)

    matchesMask = [[0, 0] for i in range(len(matches))]

    total_match = 0
    for i, match in enumerate(matches):
        if len(match) == 2:  # Ensure there are 2 matches to unpack
            m, n = match
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1, 0]
                total_match += 1

    img_res = cv2.drawMatchesKnn(
        img_object, kp_object, img_scene,
        kp_scene, matches, None,
        matchColor=[0, 255, 0], 
        singlePointColor=[255, 0, 0], 
        matchesMask=matchesMask
    )
    plt.imshow(img_res)
    plt.show()
    
def pattern_recog():
    print("Please wait...")
    train_path = 'images/train'
    person_names = os.listdir(train_path)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_list = []
    class_list = []
    for index, person_name in enumerate(person_names):
        full_name_path = train_path + '/' + person_name

        for image_path in os.listdir(full_name_path):
            full_image_path = full_name_path + '/' + image_path
            img_gray = cv2.imread(full_image_path, 0)

            detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)
            
            if(len(detected_faces) < 1):
                continue

            for face_rect in detected_faces:
                x, y, w, h = face_rect
                face_img = img_gray[y:y+w, x:x+h]

                face_list.append(face_img)
                class_list.append(index)
                
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(face_list, np.array(class_list))
    
    test_path = 'images/test'
    for image_path in os.listdir(test_path):
        full_image_path = test_path + '/' + image_path
        img_bgr = cv2.imread(full_image_path)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)

        if(len(detected_faces) < 1):
            continue

        for face_rect in detected_faces:
            x, y, w, h = face_rect
            face_img = img_gray[y:y+w, x:x+h]
        
            res, confidence = face_recognizer.predict(face_img)
            confidence = math.floor(confidence * 100) / 100

            cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 1)
            text = person_names[res] + ' ' + str(confidence) + '%'
            cv2.putText(img_bgr, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)
            cv2.imshow('res', img_bgr)
            cv2.waitKey(0)
            
def main():
    isRunning = True
    while isRunning:
        print("Welcome")
        print("1. Image Processing")
        print("2. Edge Detection")
        print("3. Shape Detection")
        print("4. Pattern Recognition")
        print("5. Exit")
        
        choice = input("Enter your choice: ")

        if choice == '1':
            image_processing()
        elif choice == '2':
            edge_detection()
        elif choice == '3':
            shape_detection()
        elif choice == '4':
            pattern_recog()
        elif choice == '5':
            isRunning = False
            print("Exiting the program.")
        else:
            print("Invalid choice. Please try again.")
        
        print("")

if __name__ == "__main__":
    main()
    