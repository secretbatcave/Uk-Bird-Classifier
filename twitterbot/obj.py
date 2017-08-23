import os
import tarfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
#from multiprocessing import Queue
import six.moves.urllib as urllib #pylint: disable=import-error

#pylint: disable=bad-whitespace,not-context-manager,too-many-arguments,too-many-locals,no-member,too-many-instance-attributes



def draw_bounding_box_on_image(image, ymin, xmin, ymax,xmax,color='red',thickness=4,display_str_list=(),use_normalized_coordinates=True):
    """Adds a bounding box to an image.

    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.

    Args:
      image: a PIL.Image object.
      ymin: ymin of bounding box.
      xmin: xmin of bounding box.
      ymax: ymax of bounding box.
      xmax: xmax of bounding box.
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list: list of strings to display in box
                        (each to be shown on its own line).
      use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)
    try:
        font = ImageFont.truetype('arial.ttf', 44)
    except IOError:
        print "NO FONTS"
        font = ImageFont.load_default()

    text_bottom = top
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        text_width +=5
        text_height +=5
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width,text_bottom)],fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),display_str,fill='black',font=font)
        text_bottom -= text_height - 2 * margin

class detectObjects(object):
    '''Wrapping the tensorflow example of object detection
    lets try and make this as puggable as possible

    takes two arguments:
    taskQ which is a python Queue object that will contain a string path to the image
    resultsQ which is also a python Queue. This will contain the results of the detection and classification
    '''

    def __init__(self, taskQ, resultsQ):
        # What model to download.
        self.MODEL_NAME     = 'ssd_mobilenet_v1_coco_11_06_2017'
        self.MODEL_FILE     = self.MODEL_NAME + '.tar.gz'
        self.DOWNLOAD_BASE  = 'http://download.tensorflow.org/models/object_detection/'

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        self.PATH_TO_CKPT   = self.MODEL_NAME + '/frozen_inference_graph.pb'
        self.PATH_TO_BIRD   = '../models/largeBirds4.pb'
        self.PATH_TO_LABELS = '../models/largeBirds4_labels.txt'

        self.taskQ          = taskQ
        self.resultsQ       = resultsQ
        self.detection_graph= ''
        self.classify_graph = ''
        #self.birdLabels     = ['squirrel', 'crow', 'wren', 'pigeon', 'cat', 'house sparrow', 'magpie', 'blackbird', 'dunnock', 'chaffinch', 'song thrush', 'robin']
        self.birdLabels     = []
        self.loadLabels()

        self.category_index =  {1: {'id': 1, 'name': u'person'},
                                2: {'id': 2, 'name': u'bicycle'},
                                3: {'id': 3, 'name': u'car'},
                                4: {'id': 4, 'name': u'motorcycle'},
                                5: {'id': 5, 'name': u'airplane'},
                                6: {'id': 6, 'name': u'bus'},
                                7: {'id': 7, 'name': u'train'},
                                8: {'id': 8, 'name': u'truck'},
                                9: {'id': 9, 'name': u'boat'},
                                10: {'id': 10, 'name': u'traffic light'},
                                11: {'id': 11, 'name': u'fire hydrant'},
                                13: {'id': 13, 'name': u'stop sign'},
                                14: {'id': 14, 'name': u'parking meter'},
                                15: {'id': 15, 'name': u'bench'},
                                16: {'id': 16, 'name': u'bird'},
                                17: {'id': 17, 'name': u'cat'},
                                18: {'id': 18, 'name': u'dog'},
                                19: {'id': 19, 'name': u'horse'},
                                20: {'id': 20, 'name': u'sheep'},
                                21: {'id': 21, 'name': u'cow'},
                                22: {'id': 22, 'name': u'elephant'},
                                23: {'id': 23, 'name': u'bear'},
                                24: {'id': 24, 'name': u'zebra'},
                                25: {'id': 25, 'name': u'giraffe'},
                                27: {'id': 27, 'name': u'backpack'},
                                28: {'id': 28, 'name': u'umbrella'},
                                31: {'id': 31, 'name': u'handbag'},
                                32: {'id': 32, 'name': u'tie'},
                                33: {'id': 33, 'name': u'suitcase'},
                                34: {'id': 34, 'name': u'frisbee'},
                                35: {'id': 35, 'name': u'skis'},
                                36: {'id': 36, 'name': u'snowboard'},
                                37: {'id': 37, 'name': u'sports ball'},
                                38: {'id': 38, 'name': u'kite'},
                                39: {'id': 39, 'name': u'baseball bat'},
                                40: {'id': 40, 'name': u'baseball glove'},
                                41: {'id': 41, 'name': u'skateboard'},
                                42: {'id': 42, 'name': u'surfboard'},
                                43: {'id': 43, 'name': u'tennis racket'},
                                44: {'id': 44, 'name': u'bottle'},
                                46: {'id': 46, 'name': u'wine glass'},
                                47: {'id': 47, 'name': u'cup'},
                                48: {'id': 48, 'name': u'fork'},
                                49: {'id': 49, 'name': u'knife'},
                                50: {'id': 50, 'name': u'spoon'},
                                51: {'id': 51, 'name': u'bowl'},
                                52: {'id': 52, 'name': u'banana'},
                                53: {'id': 53, 'name': u'apple'},
                                54: {'id': 54, 'name': u'sandwich'},
                                55: {'id': 55, 'name': u'orange'},
                                56: {'id': 56, 'name': u'broccoli'},
                                57: {'id': 57, 'name': u'carrot'},
                                58: {'id': 58, 'name': u'hot dog'},
                                59: {'id': 59, 'name': u'pizza'},
                                60: {'id': 60, 'name': u'donut'},
                                61: {'id': 61, 'name': u'cake'},
                                62: {'id': 62, 'name': u'chair'},
                                63: {'id': 63, 'name': u'couch'},
                                64: {'id': 64, 'name': u'potted plant'},
                                65: {'id': 65, 'name': u'bed'},
                                67: {'id': 67, 'name': u'dining table'},
                                70: {'id': 70, 'name': u'toilet'},
                                72: {'id': 72, 'name': u'tv'},
                                73: {'id': 73, 'name': u'laptop'},
                                74: {'id': 74, 'name': u'mouse'},
                                75: {'id': 75, 'name': u'remote'},
                                76: {'id': 76, 'name': u'keyboard'},
                                77: {'id': 77, 'name': u'cell phone'},
                                78: {'id': 78, 'name': u'microwave'},
                                79: {'id': 79, 'name': u'oven'},
                                80: {'id': 80, 'name': u'toaster'},
                                81: {'id': 81, 'name': u'sink'},
                                82: {'id': 82, 'name': u'refrigerator'},
                                84: {'id': 84, 'name': u'book'},
                                85: {'id': 85, 'name': u'clock'},
                                86: {'id': 86, 'name': u'vase'},
                                87: {'id': 87, 'name': u'scissors'},
                                88: {'id': 88, 'name': u'teddy bear'},
                                89: {'id': 89, 'name': u'hair drier'},
                                90: {'id': 90, 'name': u'toothbrush'}}



    def loadLabels(self):
        '''Give me a file and I'll put them into
        self.birdLabels
        '''
        with  open(self.PATH_TO_LABELS, 'rb') as labelFH:
            lines = labelFH.readlines()
            labels = [str(w).replace("\n", "") for w in lines]
            self.birdLabels = labels

    def downloadModel(self):
        opener = urllib.request.URLopener()
        opener.retrieve(self.DOWNLOAD_BASE + self.MODEL_FILE, self.MODEL_FILE)
        tar_file = tarfile.open(self.MODEL_FILE)
        for extractedFile in tar_file.getmembers():
            file_name = os.path.basename(extractedFile.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(extractedFile, os.getcwd())

    def marshalBoxes(self, boxes, scores, objectClasses, scoreFloor = 0.5):
        '''takes the hieght and width,
        combines the boxes and generates real pixels coords with object classes
        '''
        detectecObjects = []
        for i in range(boxes.shape[0]):
            if scores[i] > scoreFloor:
                ymin, xmin, ymax, xmax = boxes[i]
                #print "Class: \t{0}:{1}%".format(category_index[objectClasses[i]], round(scores[i]*100))
                #print "Location: \t {0},{1},{2},{3}".format(xmin, xmax,ymin,ymax)
                obj = {"class": self.category_index[objectClasses[i]], "confidence": round(scores[i]*100), "location": [ymin, xmin, ymax, xmax]}
                detectecObjects.append(obj)
        return detectecObjects



    def loadObjectGraph(self):
        print "loading detection model"
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        print "Detection Loaded."

    def loadClassifyerGraph(self):
        print "loading Classifier model"
        self.classify_graph = tf.Graph()
        with self.classify_graph.as_default():
            class_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_BIRD, 'rb') as graphFile:
                serialised_class_graph = graphFile.read()
                class_graph_def.ParseFromString(serialised_class_graph)
                tf.import_graph_def(class_graph_def, name='')
        print "Classifier Loaded"



    def load_image_into_numpy_array(self, rawImage):
        (im_width, im_height) = rawImage.size
        return np.array(rawImage.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
    def convertPilClassify(self, image):
        '''Takes a PIL image and converts it to a numpy for classify model
        '''

        imageArray = np.array(image)[:,:,0:3]
        return imageArray

    def cropImage(self, image, ymin, xmin, ymax, xmax):
        '''Takes the PIL image and returns a cropped PIL image
        '''
        im_width, im_height = image.size

        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
        croppedImage = image.crop((left,top,right,bottom))
        return croppedImage

    def classifyImage(self, image):
        with self.classify_graph.as_default():
            with tf.Session(graph=self.classify_graph) as classSess:
                softmax_tensor = classSess.graph.get_tensor_by_name('final_result:0')
                predictions = classSess.run(softmax_tensor,{'DecodeJpeg:0': image})
                predictions = np.squeeze(predictions)
                topPredictions = predictions.argsort()[-5:][::-1]
                results = []
                for predict in  topPredictions:
                    humanName = self.birdLabels[predict]
                    score = float(predictions[predict])
                    #print " {0}: {1}".format(humanName, score)
                    results.append([humanName, score])
                return results

    def makeReadable(self, image, dataStruct):
        '''Numpy things are not json serialiable
        '''
        w, h = image.size
        count = 0
        for _ in dataStruct: # ooo thats nasty
            ymin, xmin, ymax, xmax = dataStruct[count]['location']
            dataStruct[count]['location'] = float(ymin * h), float(xmin * w), float(ymax * h), float(xmax * w)
            count += 1
        return dataStruct
    def getCroppedFilePath(self, fp, count):
        '''strips out the .jpg from the file path
        and returns a file path with the correct
        number
        '''
        fileParts = fp.split('.jpg')
        newPath = "{0}-crop{1}.jpg".format(fileParts[0],count)
        return newPath



    def objectCropLoop(self, objType= None):
        '''Takes the contents of the taskQ and crops images.
        it will save as a new image with -crop01.jpg
        if given the optional objType, it will only save
        objects of that type.
        '''
        detection_graph = self.detection_graph
        print "starting detection loop"
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                print "TF session initiated"
                #for image_path in TEST_IMAGE_PATHS:
                while True:#pylint: disable=too-many-nested-blocks
                    image_path = self.taskQ.get(block=True)
                    try:
                        image = Image.open(image_path)
                    except IOError:
                        continue
                    # the array based representation of the image will be used later in order to prepare the
                    # result image with boxes and labels on it.
                    try:
                        image_np = self.load_image_into_numpy_array(image)
                    except Exception:
                        continue
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # Visualization of the results of a detection.
                    objBoundingBoxes =  self.marshalBoxes(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32))
                    if objBoundingBoxes:
                        for count, obj in enumerate(objBoundingBoxes):
                            if objType:
                                if obj['class']['name'] == objType:
                                    ymin, xmin, ymax, xmax = obj['location']
                                    croppedImage = self.cropImage(image, ymin, xmin, ymax, xmax)
                                    try:
                                        croppedImage.save(self.getCroppedFilePath(image_path, count),'JPEG', quality=75)
                                        self.resultsQ.put('{0} done'.format(image_path))
                                    except Exception, e:
                                        self.resultsQ.put('{0} not cropped: {1}'.format(image_path, e))

                            else:
                                self.resultsQ.put('{0} not cropped: {1}'.format(image_path, "no bird"))




    def objectDetectionLoop(self):
        #PATH_TO_TEST_IMAGES_DIR = 'test_images'
        #TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 5) ]
        detection_graph = self.detection_graph

        # Size, in inches, of the output images.
        print "starting detection loop"
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                print "TF session initiated"
                #for image_path in TEST_IMAGE_PATHS:
                while True:
                    image_path = self.taskQ.get(block=True)
                    image = Image.open(image_path)
                    # the array based representation of the image will be used later in order to prepare the
                    # result image with boxes and labels on it.
                    image_np = self.load_image_into_numpy_array(image)
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # Visualization of the results of a detection.
                    print image_path
                    objBoundingBoxes =  self.marshalBoxes(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32))
                    if objBoundingBoxes:
                        count = 0
                        for obj in objBoundingBoxes:
                            if obj['class']['name'] == 'bird':
                                ymin, xmin, ymax, xmax = obj['location']
                                croppedImage = self.cropImage(image, ymin, xmin, ymax, xmax)
                                pilImage = self.convertPilClassify(croppedImage)
                                result = self.classifyImage(pilImage)
                                birdPrediction = {'bird species':result}
                                objBoundingBoxes[count].update(birdPrediction)
                                #print "it is a: {0}".format(result[0])
                            count += 1
                    #self.taskQ.done()
                    self.resultsQ.put(self.makeReadable(image, objBoundingBoxes))

                    #print objBoundingBoxes
                   # print json.dumps( objBoundingBoxes)
