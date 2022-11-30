# -*- coding:utf-8 -*-
# @Time       :2022/11/28 20:05
# @AUTHOR     :XiongWanqing
# @SOFTWARE   :hybrid-images
# @task       :match face to landmarks
import dlib
import face_recognition
import math
import numpy as np
import cv2

def rect_to_bbox(rect):
    """获得人脸矩形的坐标信息"""
    # print(rect)
    x = rect[3]
    y = rect[0]
    w = rect[1] - x
    h = rect[2] - y
    return (x, y, w, h)


def face_alignment(faces):
    # 预测关键点
    predictor = dlib.shape_predictor("dat/shape_predictor_68_face_landmarks.dat")
    faces_aligned = []
    for face in faces:
        rec = dlib.rectangle(0, 0, face.shape[0], face.shape[1])
        shape = predictor(np.uint8(face), rec)
        # left eye, right eye, nose, left mouth, right mouth
        order = [36, 45, 30, 48, 54]
        for j in order:
            x = shape.part(j).x
            y = shape.part(j).y
        # 计算两眼的中心坐标
        eye_center = ((shape.part(36).x + shape.part(45).x) * 1. / 2, (shape.part(36).y + shape.part(45).y) * 1. / 2)
        dx = (shape.part(45).x - shape.part(36).x)
        dy = (shape.part(45).y - shape.part(36).y)
        # 计算角度
        angle = math.atan2(dy, dx) * 180. / math.pi
        # 计算仿射矩阵
        RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
        # 进行仿射变换，即旋转
        RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1]))
        faces_aligned.append(RotImg)
    return faces_aligned


def match(img_path):
    unknown_image = face_recognition.load_image_file(img_path)
    # 定位图片中的人脸
    face_locations = face_recognition.face_locations(unknown_image)
    # 提取人脸区域的图片并保存
    src_faces = []
    src_face_num = 0
    for (i, rect) in enumerate(face_locations):
        src_face_num = src_face_num + 1
        (x, y, w, h) = rect_to_bbox(rect)
        detect_face = unknown_image[y:y + h, x:x + w]
        src_faces.append(detect_face)
        detect_face = cv2.cvtColor(detect_face, cv2.COLOR_RGBA2BGR)
        cv2.imwrite("face_align_result/face_" + str(src_face_num) + ".jpg", detect_face)
    # 人脸对齐操作并保存
    faces_aligned = face_alignment(src_faces)
    face_num = 0
    for faces in faces_aligned:
        face_num = face_num + 1
        faces = cv2.cvtColor(faces, cv2.COLOR_RGBA2BGR)
        cv2.imwrite("face_align_result/face_align_" + str(face_num) + ".jpg", faces)
    pass


if __name__ == '__main__':
    match("selfie.jpg")
    pass

