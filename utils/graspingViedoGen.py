import cv2
import numpy as np
import os

preds=np.load('Data/preds6.npy')
num_v=np.load('Data/time_list_visual6.npy')
num_t=np.load('Data/time_list_tactile6.npy')
widths=np.load('Data/widths6.npy')
forces=np.load('Data/forces6.npy')
fps=30
path='Data/visual_6_recording/'
size=(1920,1080)

video = cv2.VideoWriter("VideoTest1.avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
font = cv2.FONT_HERSHEY_SIMPLEX
img_list=[]

for i in range(num_v[0]):
    img = cv2.imread(path + str(i) + '.jpg')
    if abs(i-num_v[0])<40:
        width = 'Width:' + str(widths[0])
        force = 'Force:' + str(forces[0])
        # img = cv2.imread(path + str(k) + '.jpg')
        img = cv2.putText(img, width, (50, 50), font, 1.2, (255, 0, 0), 2)
        img = cv2.putText(img, force, (50, 100), font, 1.2, (0, 155, 0), 2)
    if abs(i-num_v[0])<10:
        text = 'Grasping state label:' + str(preds[0])
        img = cv2.putText(img, text, (50, 150), font, 1.2, (0, 0, 0), 2)
    img_list.append(img)
for i in range(1,len(preds)):
    for k in range(num_v[i-1],num_v[i]):
        width = 'Width:' + str(widths[i-1])
        force='Force:'+str(forces[i-1])
        img=cv2.imread(path+str(k)+'.jpg')
        img = cv2.putText(img, width, (50, 50), font, 1.2, (255, 0, 0), 2)
        img = cv2.putText(img, force, (50, 100), font, 1.2, (0, 155, 0), 2)
        if abs(k-num_v[i-1])<10:
            text = 'Grasping state label:' + str(preds[i-1])
            img = cv2.putText(img, text, (50, 150), font, 1.2, (0, 0, 0), 2)
        if abs(k-num_v[i]<10):
            text = 'Grasping state label:' + str(preds[i])
            img = cv2.putText(img, text, (50, 150), font, 1.2, (0, 0, 0), 2)
        img_list.append(img)

for img in img_list:

    video.write(img)

video.release()
cv2.destroyAllWindows()