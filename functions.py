import cv2
import dlib
import numpy as np
from scipy.spatial import Delaunay

predictor_model = "./shape_predictor_68_face_landmarks.dat"


def psnr(img1, img2):
    if img1.shape != img2.shape:
        x, y = img1.shape[:2]
        res = (y, x)
        img2 = cv2.resize(img2, res, interpolation=cv2.INTER_LINEAR)
    return img2


def get_points(image):
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    try:
        detected_face = face_detector(image, 1)[0]
    except:
        print("No face detected in image {}".format(image))
    pose_landmarks = face_pose_predictor(image, detected_face)
    points = []
    for p in pose_landmarks.parts():
        points.append([p.x, p.y])
    x = image.shape[1] - 1
    y = image.shape[0] - 1
    points.append([0, 0])
    points.append([x // 2, 0])
    points.append([x, 0])
    points.append([x, y // 2])
    points.append([x, y])
    points.append([x // 2, y])
    points.append([0, y])
    points.append([0, y // 2])

    return np.array(points)


def get_triangles(points):
    return Delaunay(points).simplices


def affine_transform(input_image, input_triangle, output_triangle, size):
    warp_matrix = cv2.getAffineTransform(
        np.float32(input_triangle), np.float32(output_triangle)
    )
    output_image = cv2.warpAffine(
        input_image,
        warp_matrix,
        (size[0], size[1]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return output_image


def draw_delaunay_triangles(img, points, triangles, color=(192, 192, 192), thickness=1):
    for t in triangles:
        pts = [tuple(map(int, points[idx])) for idx in t]
        cv2.line(img, pts[0], pts[1], color, thickness)
        cv2.line(img, pts[1], pts[2], color, thickness)
        cv2.line(img, pts[2], pts[0], color, thickness)
