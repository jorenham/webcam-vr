import numpy as np
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2

SHAPE_PREDICTOR = 'shape_predictor_5_face_landmarks.dat'


def get_stream():
    # initialize the video stream and allow the cammera sensor to warmup
    print("[INFO] camera sensor warming up...")
    vs = VideoStream().start()
    time.sleep(2.0)

    return vs


def get_frame(stream: VideoStream, width):
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width
    frame = stream.read()
    return imutils.resize(frame, width=width)


def get_eye_position(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dims = np.array([frame.shape[1], frame.shape[0]])

    # detect faces in the grayscale frame
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR)
    rects = detector(gray, 0)

    # nothing detected
    if not rects:
        return None

    # only use first detected first face
    rect = rects[0]

    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # use the mean of the first and second two points to find the pupillary
    # distance in relative coords
    pupils = np.vstack((
        (shape[0, :] + shape[1, :]) / 2,
        (shape[2, :] + shape[3, :]) / 2,
    ))
    pupils /= dims
    pupillary_distance = np.abs(np.diff(pupils))

    # find x, y position of eye center
    position = (pupils[0, :] + pupils[1, :]) / 2

    # append z [0, 1] coordinate based on pd
    position = np.append(position, pupillary_distance[1])

    return position


def show_frame_with_position(position, frame):
    # draw virtual position
    position_pixels = (
        int((1 - position[0]) * frame.shape[1]),
        int((1 - position[1]) * frame.shape[0])
    )
    size = int((frame.shape[1] / 10) * position[2])
    color = (0, 255, 0)
    cv2.line(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)), position_pixels, color)
    cv2.circle(frame, position_pixels, size, color, -1)

    # show the frame
    cv2.imshow("Frame", frame)


if __name__ == '__main__':
    vs = get_stream()
    try:
        prev = None
        while True:
            f = get_frame(stream=vs, width=1200)
            pos = get_eye_position(f)

            # use previous value if no face is detected
            if pos is None and prev is not None:
                pos = prev

            show_frame_with_position(pos, f)
    except KeyboardInterrupt:
        pass
    finally:
        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()
