import cv2
import os

import numpy as np

root = "Test"
root1 = "testResult1"
root2 = "testResult2"

filenames1 = os.listdir(root1)
filenames2 = os.listdir(root2)

for i in range(len(os.listdir(root))):
    length = len(str(i))
    for filename1 in filenames1:
        # print(filename1)
        # print(filename1[13:13 + length])
        # print(filename1[13+length+1:13+length+2])
        if filename1[13:13+length] == str(i) and filename1[13+length:13+length+1] == "_" and filename1[13+length:13+length+4] != "_seg":
            origin = cv2.imread(os.path.join(root1, filename1))
    for filename1 in filenames1:
        if filename1[13:13+length] == str(i) and filename1[13+length:13+length+4] == "_seg":
            img1 = cv2.imread(os.path.join(root1, filename1))
            img1 = cv2.resize(img1, (origin.shape[1], origin.shape[0]))
    for filename2 in filenames2:
        if filename2[13:13+length] == str(i) and filename2[13+length:13+length+4] == "_seg":
            img2 = cv2.imread(os.path.join(root2, filename2))
            img2 = cv2.resize(img2, (origin.shape[1], origin.shape[0]))

    res = np.hstack([origin, img1, img2])
    # cv2.imshow('1', res)
    # cv2.waitKey()
    os.makedirs("Result", exist_ok=True)
    cv2.imwrite(os.path.join("Result", str(i) + ".jpg"), res)

