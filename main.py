"""
Human facial landmark detector based on Convolutional Neural Network.
https://towardsdatascience.com/real-time-head-pose-estimation-in-python-e52db1bc606a
"""

import numpy as np
import tensorflow as tf
from threading import Thread
import cv2
import math
import logging
import datetime

logger = logging.getLogger('Beep-PCR-Test-Kit')
logger.setLevel(logging.DEBUG)
logging.basicConfig(format="%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s")

# https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
# https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
# https://stackoverflow.com/questions/58293187/opencv-real-time-streaming-video-capture-is-slow-how-to-drop-frames-or-get-sync

SPIT = False
START_TIME_CALIBRATE = None
END_TIME_CALIBRATE = None
START_TIME_LOOK_UP = None
END_TIME_LOOK_UP = None
MARGIN_OF_ERR = 5
DURATION_TO_CALIBRATE = 5
DURATION_TO_LOOK_UP = 5
NUMBER_OF_ANGLES_TO_CALIBRATE = 5
DURATION_TO_STABILIZE = 5
stabilize_position = 0
calibrate_angle = []
is_recording = False
DURATION_TO_TILT_HEAD_UP = 5


class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


class FaceDetector:
    """
    Detect human face from image, use Caffe Model of OpenCV DNN. See more:
    https://towardsdatascience.com/face-detection-models-which-to-use-and-why-d263e82c302c
    """

    def __init__(self,
                 dnn_proto_text='models/deploy.prototxt',
                 dnn_model='models/res10_300x300_ssd_iter_140000.caffemodel'):

        """
        Initialization
        face_detector = FaceDetector()
        """

        self.face_net = cv2.dnn.readNetFromCaffe(dnn_proto_text, dnn_model)
        self.detection_result = None

    def get_faceboxes(self, image, threshold=0.5):

        """
        Get the bounding box of faces in image using dnn.
        """

        rows, cols, channels = image.shape

        confidences = []
        face_detector_faceboxes = []

        """
        Set input of neural network to be the image caught by the camera.
        """

        self.face_net.setInput(
            cv2.dnn.blobFromImage(
                image, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False
            )
        )

        """
        I'm guessing forward is to get the neural network to start finding the face, returning detections 
        if it found the face
        """

        detections = self.face_net.forward()

        """
        Detection is a 4 dimensional array. Most outer and second-most outer has one child element
        For each result, index 0 and 1, don't really know what they return
        index 2 is the confidence
        index 3 - 6 is the box points
        https://answers.opencv.org/question/208419/can-someone-explain-the-output-of-forward-in-dnn-module/
        """

        for result in detections[0, 0, :, :]:
            confidence = result[2]
            if confidence > threshold:
                x_left_bottom = int(result[3] * cols)
                y_left_bottom = int(result[4] * rows)
                x_right_top = int(result[5] * cols)
                y_right_top = int(result[6] * rows)

                """
                Storing confidence and box points into array, don't have to worry about updating and clearning
                the appended values since confidences and and face_detector_faceboxes are always initialized
                as empty arrays
                """

                confidences.append(confidence)
                face_detector_faceboxes.append(
                    [x_left_bottom, y_left_bottom, x_right_top, y_right_top])

        self.detection_result = [face_detector_faceboxes, confidences]

        return confidences, face_detector_faceboxes

    def draw_all_result(self, image):

        """
        Pre-requisite: run get_faceboxes() first if not self.detection_result will be null)
        Draw the detection result on image
        """

        try:
            conf = self.detection_result[1][0]
            facebox = self.detection_result[0][0]

            """
            Rectangle for the face
            """

            cv2.rectangle(image, (facebox[0], facebox[1]), (facebox[2], facebox[3]), (0, 255, 0))
            label = "face: %.4f" % conf
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            """
            Rectangle for the label
            """

            cv2.rectangle(image, (facebox[0], facebox[1] - label_size[1]),
                          (facebox[0] + label_size[0],
                           facebox[1] + base_line),
                          (0, 255, 0), cv2.FILLED)

            cv2.putText(image, label, (facebox[0], facebox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        except IndexError:
            logger.error("Cannot detect face!!!!")


class MarkDetector:
    """
    Facial landmark detector by Convolutional Neural Network
    """

    def __init__(self, saved_model='models/pose_model'):

        """
        Initialization
        mark_detector = MarkDetector()
        """

        # A face detector is required for mark detection.
        self.face_detector = FaceDetector()

        self.cnn_input_size = 128
        self.marks = None

        # Restore model from the saved_model file.
        self.model = tf.saved_model.load(saved_model)

    @staticmethod
    def draw_box(image, boxes, box_color=(255, 255, 255)):

        """
        Draw square boxes on image
        """

        for box in boxes:
            cv2.rectangle(image,
                          (box[0], box[1]),
                          (box[2], box[3]),
                          box_color,
                          3)

    @staticmethod
    def move_box(box, offset):

        """
        Move the box to direction specified by vector offset
        box would have 4 parameters
        offset would have 2 parameters
        """

        left_x = box[0] + offset[0]
        top_y = box[1] + offset[1]
        right_x = box[2] + offset[0]
        bottom_y = box[3] + offset[1]
        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def get_square_box(box):

        """
        Get a square box out of the given box, by expanding it.
        """

        left_x = box[0]
        top_y = box[1]
        right_x = box[2]
        bottom_y = box[3]

        box_width = right_x - left_x
        box_height = bottom_y - top_y

        # Check if box is already a square. If not, make it a square.
        diff = box_height - box_width
        delta = int(abs(diff) / 2)

        if diff == 0:  # Already a square.
            return box
        elif diff > 0:  # Height > width, a slim box.
            left_x -= delta
            right_x += delta
            if diff % 2 == 1:
                right_x += 1
        else:  # Width > height, a short box.
            top_y -= delta
            bottom_y += delta
            if diff % 2 == 1:
                bottom_y += 1

        # Make sure box is always square.
        assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def box_in_image(box, image):
        """Check if the box is in image"""
        rows = image.shape[0]
        cols = image.shape[1]
        return box[0] >= 0 and box[1] >= 0 and box[2] <= cols and box[3] <= rows

    def extract_cnn_facebox(self, image):

        """
        Extract face area from image, in a squared-form for CNN
        """

        _, raw_boxes = self.face_detector.get_faceboxes(image=image, threshold=0.5)
        a = []

        """
        Process the box to make it square and perform any offsets
        Ensure that the processed box is still within the image frame
        """

        for box in raw_boxes:
            # Move box down.
            # diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
            offset_y = int(abs((box[3] - box[1]) * 0.1))
            box_moved = self.move_box(box, [0, offset_y])

            # Make box square.
            the_facebox = self.get_square_box(box_moved)

            # Draw out squared-box
            # self.draw_box(image, [the_facebox], (0, 255, 0))

            if self.box_in_image(the_facebox, image):
                a.append(the_facebox)

        """
        Draw the results on the image
        """

        # print(a)

        # self.face_detector.draw_all_result(image=image)

        # How draw_box() works
        # self.draw_box(image, a)

        # if you want to see how move_box() works using the squared-box
        # moved_box = self.move_box(a[0], [50, 50])
        # self.draw_box(image,  [moved_box])

        # if you want to see, original raw_boxes
        # self.draw_box(image, raw_boxes, (255, 0, 0))

        return a

    def detect_marks(self, image_np):

        """
        Detect facial landmarks from image, using CNN
        """

        # Actual detection.
        predictions = self.model.signatures["predict"](tf.constant(image_np, dtype=tf.uint8))

        # Convert predictions to landmarks.
        image_marks = np.array(predictions['output']).flatten()[:136]
        image_marks = np.reshape(image_marks, (-1, 2))

        return image_marks

    @staticmethod
    def draw_marks(image, mark_points, color=(255, 255, 255)):

        """
        Draw mark points on image
        """

        for mark in mark_points:
            cv2.circle(image, (int(mark[0]), int(mark[1])), 2, color, -1, cv2.LINE_AA)


def draw_annotation_box(img_to_draw_box_on, rec, x_rotation_vector, x_translation_vector, x_camera_matrix,
                        color=(255, 255, 0), line_width=2):
    def helper_function(array, the_size, depth):
        array.append((-the_size, -the_size, depth))
        array.append((-the_size, the_size, depth))
        array.append((the_size, the_size, depth))
        array.append((the_size, -the_size, depth))
        array.append((-the_size, -the_size, depth))
        return array

    """
    Draw a 3D box as annotation of pose
    """

    point_3d = []

    # 4 row 1 col 0 numpy array
    distance_coefficients = np.zeros((4, 1))

    rear_size = 1
    rear_depth = 0
    point_3d = helper_function(point_3d, rear_size, rear_depth)

    front_size = img_to_draw_box_on.shape[1]
    front_depth = front_size * 2
    point_3d = helper_function(point_3d, front_size, front_depth)

    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      x_rotation_vector,
                                      x_translation_vector,
                                      x_camera_matrix,
                                      distance_coefficients)

    point_2d = np.int32(point_2d.reshape(-1, 2))

    # # Draw all the lines

    k = (point_2d[5] + point_2d[8]) // 2

    cv2.polylines(img_to_draw_box_on, [point_2d[5:10]], True, color, line_width, cv2.LINE_AA)
    cv2.line(img_to_draw_box_on, tuple(rec[0]), tuple(point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img_to_draw_box_on, tuple(rec[1]), tuple(point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img_to_draw_box_on, tuple(rec[3]), tuple(point_2d[8]), color, line_width, cv2.LINE_AA)
    cv2.line(img_to_draw_box_on, tuple(rec[2]), tuple(point_2d[9]), color, line_width, cv2.LINE_AA)

    return point_2d[2], k


# MarkDetector uses FaceDetector since you first need to detect the first
mark_detector = MarkDetector()

# With Threading
vs = WebcamVideoStream(src=0).start()

img = vs.read()

size = img.shape
font = cv2.FONT_HERSHEY_SIMPLEX

# 3D model points.
model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corne
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
])

focal_length = size[1]
center = (size[1] / 2, size[0] / 2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]],
     [0, focal_length, center[1]],
     [0, 0, 1]], dtype="double"
)

# With Threading
while True:

    img = vs.read()

    if img is not ():

        if True:
            faceboxes = mark_detector.extract_cnn_facebox(img)
            for facebox in faceboxes:

                face_img = img[facebox[1]: facebox[3], facebox[0]: facebox[2]]
                face_img = cv2.resize(face_img, (256, 256))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                marks = mark_detector.detect_marks([face_img])
                marks *= (facebox[2] - facebox[0])
                marks[:, 0] += facebox[0]
                marks[:, 1] += facebox[1]
                shape = marks.astype(np.uint)

                # Draw facial landmarks on face
                mark_detector.draw_marks(img, marks, color=(0, 255, 0))

                image_points = np.array([
                    shape[30],  # Nose tip
                    shape[8],  # Chin
                    shape[36],  # Left eye left corner
                    shape[45],  # Right eye right corner
                    shape[48],  # Left Mouth corner
                    shape[54]  # Right mouth corner
                ], dtype="double")
                dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
                (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                              dist_coeffs, flags=cv2.SOLVEPNP_UPNP)

                # Project a 3D point (0, 0, 1000.0) onto the image plane.
                # We use this to draw a line sticking out of the nose

                (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                                 translation_vector, camera_matrix, dist_coeffs)

                p1 = (int(image_points[0][0]), int(image_points[0][1]))
                p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                cv2.rectangle(img, (facebox[0], facebox[1]), (facebox[2], facebox[3]), (255, 255, 0), 2)
                # draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix)

                x1, x2 = draw_annotation_box(img, (
                                                (facebox[0], facebox[1]),
                                                (facebox[2], facebox[1]),
                                                (facebox[0], facebox[3]),
                                                (facebox[2], facebox[3])
                                                    ), rotation_vector, translation_vector, camera_matrix)

                try:
                    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
                    ang1 = int(math.degrees(math.atan(m)))
                except:
                    print('div by zero error')
                    ang1 = 90

                try:
                    m = (x2[1] - x1[1]) / (x2[0] - x1[0])
                    ang2 = int(math.degrees(math.atan(-1 / m)))
                except:
                    print('div by zero error')
                    ang2 = 90

                # this is for eyes, node, mouth and chin
                for p in image_points:
                    cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

                # this is for... everything lol? over-rights the previous plots
                # for (x, y) in shape:
                #     cv2.circle(img, (x, y), 4, (255, 255, 0), -1)

                # this is for line1
                cv2.line(img, p1, p2, (0, 255, 255), 2)
                cv2.putText(img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
                cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
                cv2.putText(img, str(p2), p2, font, 1, (0, 255, 255), 1)

                # this is for line2
                # cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)
                # cv2.putText(img, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)

                logger.info(f'Angle: {abs(ang1)}')

                if SPIT:
                    # Wait 5 seconds....
                    # SPIT = False
                    logger.info("Think of a work around")
                    pass
                else:
                    if END_TIME_CALIBRATE is None:
                        END_TIME_CALIBRATE = datetime.datetime.now() + datetime.timedelta(seconds=DURATION_TO_CALIBRATE)

                    START_TIME_CALIBRATE = datetime.datetime.now()

                    if START_TIME_CALIBRATE >= END_TIME_CALIBRATE:
                        # proceed to calibrate
                        logger.info("proceed to calibrate")
                        is_recording = True
                        logger.info(np.mean(calibrate_angle))
                        if abs(ang1) <= np.mean(calibrate_angle) + 20:
                            logger.warning("Tilt your head back more!")
                            END_TIME_LOOK_UP = None
                        else:
                            START_TIME_LOOK_UP = datetime.datetime.now()
                            if END_TIME_LOOK_UP is None:
                                END_TIME_LOOK_UP = datetime.datetime.now() + datetime.timedelta(
                                    seconds=DURATION_TO_LOOK_UP)
                            if START_TIME_LOOK_UP >= END_TIME_LOOK_UP:
                                logger.info("Alright... pls spit")
                                END_TIME_LOOK_UP = None
                                SPIT = True
                            else:
                                logger.info("Hold it up for 5 seconds")
                    else:
                        if len(calibrate_angle) == 0 or \
                                np.mean(calibrate_angle) - 5 <= ang1 <= np.mean(calibrate_angle) + 5:
                            calibrate_angle.append(abs(ang1))
                        else:
                            logger.warning("You are tilting your head too much")
                            logger.warning("Resetting calibrated angles")
                            calibrate_angle = []
                            END_TIME_CALIBRATE = None

        cv2.imshow("Normal Frame", cv2.flip(img, 1))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break


cv2.destroyAllWindows()
vs.stream.release()
vs.stop()
