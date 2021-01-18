import cv2
import imutils
import time

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
bgr = [148, 145, 145]

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        count, FPS = 0, 0
        start = time.time()

        success, crop = self.video.read()

        frame = imutils.resize(crop, width=600, height=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        if len(face_rects):
            for (x, y, w, h) in face_rects:
                crop = frame[y - 50: y + h + 50, x - 50: x + w + 50]
                crop = cv2.resize(crop, (200, 250))
                crop = cv2.copyMakeBorder(crop, 100, 100, 150, 150, cv2.BORDER_CONSTANT, value=bgr)
                break
        else:
            crop = cv2.resize(crop, (200, 250))
            crop = cv2.copyMakeBorder(crop, 100, 100, 150, 150, cv2.BORDER_CONSTANT, value=bgr)

        FPS = round(1 / (time.time() - start))
        cv2.putText(crop, "FPS = {}".format(FPS), (24, 24), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 24, 248), 2)

        ret, jpeg = cv2.imencode('.jpg', crop)
        return jpeg.tobytes()
