import cv2
import numpy as np
import dlib
from PIL import Image


net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

categories = {
    0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle',
    6: 'bus', 7: 'car', 8: 'cat',9: 'chair', 10: 'cow',
    11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person',
    16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'
}

video = cv2.VideoCapture('yourVideo.mp4')

colors = np.random.randint(0, 255, size=(len(categories), 3)).tolist()

trackers = []
labels = []
idxs=[]
#把目標轉成模型需要的blob格式
#調整成固定尺寸，需調成300*300
#0.007843 => scalefactor:圖像各通道數值的縮放比例
#127.5，用於各通道減去的值，以降低光照的影響
#需要透過這個過程把圖片弄成MobileNetSSD的尺寸

# write video
ret,frame = video.read()
(h, w, c) = frame.shape
fps = video.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
output = cv2.VideoWriter('result.mp4', fourcc, 20.0, (int(w*0.7), int(h*0.7)))

while True:
    ret,frame = video.read()
    if frame is None:
        break

    (h, w, c) = frame.shape

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), (127.5,127.5,127.5))
    net.setInput(blob)
    detections = net.forward()

    if len(detections) > 0:
        for i in np.arange(0, detections.shape[2]):
            #取出信心程度
            confidence = detections[0, 0, i, 2]

            if confidence >0.4:
                idx = int(detections[0, 0, i, 1])
                #取出座標點，但是這裡顯示的是比例，將原本的圖片尺寸做還原
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                #座標沒有小數
                (startX, startY, endX, endY) = box.astype("int")

                cv2.rectangle(frame, (startX, startY), (endX, endY), colors[idx], 2)
                label = "{}: {:.2f}%".format(categories[idx], confidence * 100)
                #如果上面沒空間就放下面一點
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

    frame = cv2.resize(frame,(int(w*0.7),int(h*0.7)),cv2.INTER_AREA)

    cv2.imshow("Output", frame)
    output.write(frame)

    k = cv2.waitKey(20)
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()

