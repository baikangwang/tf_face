import sys
import threading
import os
from Queue import Queue
import imghdr

NUM_THREADS = 1


class MetaWriterThread(threading.Thread):
    def __init__(self, queue, output_dir, batch_index):
        super(MetaWriterThread, self).__init__()
        self.queue = queue
        self.daemon = True
        self.output_dir = output_dir
        self.batch_index = batch_index

    def run(self):
        file_path = os.path.join(
            self.output_dir, ("clean-manifest_%s.txt" % self.batch_index))
        stream = open(file_path, 'w')
        while True:
            # print "main %d" % self.queue.qsize()
            stream.write(self.queue.get() + "\n")
            self.queue.task_done()
        stream.close()


class OpencvThread(threading.Thread):
    def __init__(self, image_queue, print_queue, prepend_dir):
        super(OpencvThread, self).__init__()
        self.image_queue = image_queue
        self.print_queue = print_queue
        self.prepend_dir = prepend_dir
        self.daemon = True

    def run(self):
        while not self.image_queue.empty():
            try:
                next_file = self.image_queue.get()
                filename = os.path.join(self.prepend_dir, next_file)

                # check if file dose still exist
                try:
                    if not os.path.isfile(filename):
                        self.clean_image(filename, "Not exists", False)
                        continue
                except Exception as e:
                    self.clean_image(filename, str(e), False)
                    continue

                # check file size and skip if size ==0
                # try:
                #     file_size = os.path.getsize(filename)
                #     print(file_size)
                #     if file_size == 0:
                #         self.clean_image(filename, 'Empty image')
                #         continue
                # except Exception as e:
                #     # when the path dosen't exist or access deny
                #     self.clean_image(filename, str(e), False)
                #     continue

                # check if image is valid
                try:
                    valid = imghdr.what(filename)
                    if valid is None:
                        self.clean_image(filename, msg='Invalid image')
                except Exception as e:
                    self.clean_image(filename, str(e), False)

            except Exception as e:
                self.clean_image(filename, str(e), False)

    def clean_image(self, filename, msg='', toberemoved=True):
        print "%s|%s" % (filename, msg)
        self.print_queue.put("%s|%s" % (filename, msg))
        # print self.print_queue.qsize()
        if toberemoved:
            os.remove(filename)
            print "%s|Removed" % (filename)

        self.image_queue.task_done()


def read_file(file_path):
    stream = open(file_path)
    queue = Queue()

    for line in stream:
        if not line.startswith('#'):
            tokens = line.split('|')
            queue.put(tokens[0].rstrip('\n'))

    stream.close()
    return queue


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: clean_faces.py <input_file> [prepend_dir] [batch_index]"
        exit(0)

    image_queue = read_file(sys.argv[1])
    prepend_dir = "./"
    batch_index = "0"

    if len(sys.argv) >= 3:
        prepend_dir = sys.argv[2]

    if len(sys.argv) >= 4:
        batch_index = sys.argv[3]

    print_queue = Queue()

    for i in range(NUM_THREADS):
        t = OpencvThread(image_queue, print_queue, prepend_dir)
        t.start()

    t = MetaWriterThread(
        print_queue, os.path.dirname(sys.argv[1]), batch_index)
    t.start()

    print_queue.join()
    image_queue.join()
