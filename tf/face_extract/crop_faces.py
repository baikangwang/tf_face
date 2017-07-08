import numpy
import cv2
import sys
import threading
import io
import re
import os
from Queue import Queue
# from PIL import Image
import imghdr

NUM_THREADS = 1
CASCPATH='./tf/face_extract/haarcascade_frontalface_faceTracker.xml'

class MetaWriterThread(threading.Thread):
    def __init__(self, queue, output_dir, batch_index):
        super(MetaWriterThread, self).__init__()
        self.queue = queue
        self.daemon = True
        self.output_dir = output_dir
        self.batch_index = batch_index

    def run(self):
        file_path = os.path.join(self.output_dir, ("opencv-manifest_%s.txt" % self.batch_index))
        f = open(file_path, 'w')
        while True:
            f.write(self.queue.get() + "\n")
            self.queue.task_done()
        f.close()

class OpencvThread(threading.Thread):
    def __init__(self, image_queue, print_queue, prepend_dir):
        super(OpencvThread, self).__init__()
        self.image_queue = image_queue
        self.print_queue = print_queue
        self.prepend_dir = prepend_dir
        self.daemon = True
        # self.client = vision.Client.from_service_account_json(os.path.join(sys.path[0], './vapi-acct.json'), PROJECT_ID)

    def run(self):
        while True:
            try:
                next_file = self.image_queue.get()
                filename = os.path.join(self.prepend_dir, next_file)
                print("[%s] Info: Opening file %s" % (self.ident, filename))

                # check file size and skip if size ==0
                #try:
                #    fileSize = os.path.getsize(filename)
                #    if fileSize==0:
                #        print("[%s] Warn: Invalid file %s" % (self.ident, filename))
                #        continue
                #except:
                #    # when the path dosen't exist or access deny
                #    print("[%s] Warn: Invalid file %s" % (self.ident, filename))
                #    continue

                # check if image is valid
                try:
                    valid=imghdr.what(filename)
                    if valid==None:
                        print("[%s] Warn: Invalid file %s" % (self.ident, filename))
                        continue
                except:
                    print("[%s] Warn: Invalid file %s" % (self.ident, filename))
                    continue

                faceCascade = cv2.CascadeClassifier(CASCPATH)
                image = cv2.imread(filename,1)
                gray = cv2.imread(filename,0)

                faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.4,
                    minNeighbors=4,
                    minSize=(30,30)
                )

                if len(faces) > 0:
                    for i,(x,y,w,h) in enumerate(faces):
                        left = x
                        top = y
                        right = x+w
                        bottom = y+h

                        # crop image if all required verticies exists
                        if left and top and right and bottom:
                            file_dir = os.path.dirname(next_file)
                            base_file = os.path.basename(next_file)
                            crop_dir = os.path.join(file_dir, 'crop')

                            # create the output directory if it doesn't exist
                            if not os.path.exists(crop_dir):
                                print("[%s] Info: mkdir %s" % (self.ident, crop_dir))
                                os.makedirs(crop_dir)

                            cropped = image[top:bottom,left:right]
                            cropped = cv2.resize(src=cropped,dsize=(96,96),interpolation=cv2.INTER_CUBIC)
                            out_file = os.path.join(crop_dir, "crop_%d_%s" % (i,base_file))

                            print("[%s] Info: Saving file %s" % (self.ident, out_file))
                            cv2.imwrite(out_file,cropped)

                            # try and get face angle information
                            # angles = faces[0].angles
                            # pan = ''
                            # roll = ''
                            # tilt = ''

                            #if angles != None:
                            #    pan = angles.pan
                            #    roll = angles.roll
                            #    tilt = angles.tilt

                            # write image details to data file
                            line = "%s|%s|%s|%s|%s" % (out_file, left, top, right,bottom)
                            self.print_queue.put(line)

                        else:
                            print("[%s] Error: Incomplete coordinates for %s" % (self.ident, filename))
                else:
                    print("[%s] Error: No face detected for %s" % (self.ident, filename))

            except Exception as e:
                print("[%s] Error: Exception occurred for %s: %s" % (self.ident, filename, str(e)))

            self.image_queue.task_done()

    def crop_image(self, filename):
        print("[%s] Cropping %s" % (self.ident, filename))

def read_file(file_path):
    f = open(file_path)
    queue = Queue()

    for line in f:
        if not line.startswith('#'):
            tokens = line.split('|')
            queue.put(tokens[0].rstrip('\n'))

    f.close()
    return queue

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: crop_faces.py <input_file> [prepend_dir] [batch_index]")
        exit(0)

    image_queue = read_file(sys.argv[1])
    prepend_dir = "./"
    batch_index = "0"

    if len(sys.argv) >= 3:
        prepend_dir = sys.argv[2]

    if len(sys.argv) >= 4:
        batch_index=sys.argv[3]

    print_queue = Queue()
    print_queue.put("#file|pan|roll|tilt")

    for i in range(NUM_THREADS):
        t = OpencvThread(image_queue, print_queue, prepend_dir)
        t.start()

    t = MetaWriterThread(print_queue, os.path.dirname(sys.argv[1]), batch_index)
    t.start()

    image_queue.join()
    print_queue.join()
