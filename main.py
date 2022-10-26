import torch
import time
import cv2
import numpy as np
from multiprocessing import Process, Queue, Pipe


def do_detection(Frames_queue, Results_queue, model):
    while True:
        if Frames_queue.qsize() > 0:
            frame = Frames_queue.get()
            results = model(frame, size=320)  # pass the image through our model
            Results_queue.put(results)


def show_results(Results_queue, model):
    while True:
        if Results_queue.qsize() > 0:
            results = Results_queue.get()
            im_list = results.render()
            im = im_list[0]
            cv2.imshow("out", im)
            if cv2.waitKey(40) == 27:
                break
        #


if __name__ == '__main__':

    model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local',
                           device='cpu')  # local repo
    cap = cv2.VideoCapture(0)
    Frames_queue = Queue()
    Results_queue = Queue()
    p1 = Process(target=do_detection, args=(Frames_queue, Results_queue, model))
    p2 = Process(target=show_results, args=(Results_queue, model))
    p1.start()
    p2.start()
    timor = time.time()
    while cap.isOpened():
        ret, frame1 = cap.read()
        if not ret:
            break
        Frames_queue.put(frame1)
    while True:
        if Frames_queue.qsize() == 0 and Results_queue.qsize() == 0:
            p1.terminate()
            p2.terminate()
            break

    cv2.destroyAllWindows()
    cap.release()
