import numpy as np
import cv2
import sys
from PyQt5 import QtWidgets
from progress.bar import IncrementalBar
import time
from functions import psnr, affine_transform, get_points, get_triangles, draw_delaunay_triangles
from data_from_json import ADD_PRCNT_PPT_X, ADD_PRCNT_PPT_Y_DOWN, ADD_PRCNT_PPT_Y_UPPER, DEFAULT_ALPHA_VALUE, \
    DEFAULT_ALPHA_VALUE_FLG, COUNT_OF_OUT_MORPHS
from Ui import Ui

if __name__ == "__main__":
    print()
    alpha = DEFAULT_ALPHA_VALUE
    if not DEFAULT_ALPHA_VALUE_FLG:
        alpha = (
            Ui().input_alpha()
        )  # чем ближе к 1, тем больше итоговое изображение будет похоже на вторую исходную
    if DEFAULT_ALPHA_VALUE:
        print("---------------------------------------------\n"
              "Значение alpha = 0.5 (по умолчанию)")
    else:
        print("---------------------------------------------\n"
              f"Значение alpha = {alpha} (пользовательское)")

    app = QtWidgets.QApplication(sys.argv)
    print(
        "\n"
        "---------------------------------------------\n"
        "Выберите первое изображение...\n"
    )
    img1_name = Ui().filename1_ButtonPressed()
    print(img1_name)

    img1_general = cv2.imread(img1_name)

    print()
    bar = IncrementalBar("Определение лица на первом фото:", max=5)
    bar.next()
    points1 = get_points(img1_general)
    bar.next()
    img1_show = np.zeros(img1_general.shape, np.uint8)
    bar.next()
    img1_show = img1_general.copy()
    bar.next()
    for idx, point in enumerate(points1):
        pos = (point[0], point[1])
        cv2.putText(
            img1_show,
            str(idx),
            pos,
            fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            fontScale=0.1,
            color=(0, 255, 0),
        )
    bar.next()
    bar.finish()

    # cv2.imwrite(Ui().save_file(0), img1_show)

    points1_list = [[int(y) for y in x] for x in points1]
    print()

    for num in range(COUNT_OF_OUT_MORPHS):
        print(f"Выберите второе изображение ({num + 1})...")
        img2_name = Ui().filename2_ButtonPressed_2(num)
        img2 = cv2.imread(img2_name)

        bar = IncrementalBar(f"Определение лица на втором фото ({num + 1}):", max=5)
        bar.next()
        points2 = get_points(img2)
        bar.next()
        img2_show = np.zeros(img2.shape, np.uint8)
        bar.next()
        img2_show = img2.copy()
        bar.next()
        for idx, point in enumerate(points2):
            pos = (point[0], point[1])
            cv2.putText(
                img2_show,
                str(idx),
                pos,
                fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                fontScale=0.1,
                color=(0, 255, 0),
            )
        bar.next()
        bar.finish()

        points2_list = [[int(y) for y in x] for x in points2]
        min_first_point1 = 0
        max_first_point1 = max([i[0] for i in points1_list])
        min_second_point1 = 0
        max_second_point1 = max([i[1] for i in points1_list])
        min_first_point2 = 0
        max_first_point2 = max([i[0] for i in points2_list])
        min_second_point2 = 0
        max_second_point2 = max([i[1] for i in points2_list])

        first_image_left_up_point_x = max_first_point1
        first_image_left_up_point_y = max_second_point1
        first_image_right_down_point_x = min_first_point1
        first_image_right_down_point_y = min_second_point1
        second_image_left_up_point_x = max_first_point2
        second_image_left_up_point_y = max_second_point2
        second_image_right_down_point_x = min_first_point2
        second_image_right_down_point_y = min_second_point2

        for i in points1_list:  # координаты точек face_box'а первого изображения
            if (
                    i[0] != 0
                    and i[0] != max_first_point1
                    and i[1] != 0
                    and i[1] != max_second_point1
            ):
                first_image_left_up_point_x = min(first_image_left_up_point_x, i[0])
                first_image_left_up_point_y = min(first_image_left_up_point_y, i[1])
                first_image_right_down_point_x = max(first_image_right_down_point_x, i[0])
                first_image_right_down_point_y = max(first_image_right_down_point_y, i[1])

        for i in points2_list:  # координаты точек face_box'а второго изображения
            if (
                    i[0] != 0
                    and i[0] != max_first_point2
                    and i[1] != 0
                    and i[1] != max_second_point2
            ):
                second_image_left_up_point_x = min(second_image_left_up_point_x, i[0])
                second_image_left_up_point_y = min(second_image_left_up_point_y, i[1])
                second_image_right_down_point_x = max(second_image_right_down_point_x, i[0])
                second_image_right_down_point_y = max(second_image_right_down_point_y, i[1])

        img1 = img1_general[
               max(
                   first_image_left_up_point_y
                   - int(
                       (first_image_right_down_point_y - first_image_left_up_point_y)
                       * ADD_PRCNT_PPT_Y_UPPER
                   ),
                   0,
               ): first_image_right_down_point_y
                  + int(
                   (first_image_right_down_point_y - first_image_left_up_point_y)
                   * ADD_PRCNT_PPT_Y_DOWN
               ),
               max(
                   first_image_left_up_point_x
                   - int(
                       (first_image_right_down_point_x - first_image_left_up_point_x)
                       * ADD_PRCNT_PPT_X
                   ),
                   0,
               ): first_image_right_down_point_x
                  + int(
                   (first_image_right_down_point_x - first_image_left_up_point_x)
                   * ADD_PRCNT_PPT_X
               ),
               ].copy()

        img2 = img2[
               max(
                   second_image_left_up_point_y
                   - int(
                       (second_image_right_down_point_y - second_image_left_up_point_y)
                       * ADD_PRCNT_PPT_Y_UPPER
                   ),
                   0,
               ): second_image_right_down_point_y
                  + int(
                   (second_image_right_down_point_y - second_image_left_up_point_y)
                   * ADD_PRCNT_PPT_Y_DOWN
               ),
               max(
                   second_image_left_up_point_x
                   - int(
                       (second_image_right_down_point_x - second_image_left_up_point_x)
                       * ADD_PRCNT_PPT_X
                   ),
                   0,
               ): second_image_right_down_point_x
                  + int(
                   (second_image_right_down_point_x - second_image_left_up_point_x)
                   * ADD_PRCNT_PPT_X
               ),
               ].copy()

        if img1.shape > img2.shape:
            img2 = psnr(img1, img2)
        else:
            img1 = psnr(img2, img1)

        # print(1)
        # cv2.imwrite(Ui().save_file(num), img1) #
        # cv2.imwrite(Ui().save_file(num), img2) #
        # print("1 end")

        bar = IncrementalBar("Составление треугольников:", max=6)
        bar.next()
        points1 = get_points(img1)
        img1_show = np.zeros(img1.shape, np.uint8)
        img1_show = img1.copy()
        bar.next()
        for idx, point in enumerate(points1):
            pos = (point[0], point[1])
            cv2.putText(
                img1_show,
                str(idx),
                pos,
                fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                fontScale=0.1,
                color=(0, 255, 0),
            )
        bar.next()

        cv2.imwrite(Ui().save_file(0), img1_show)

        points2 = get_points(img2)
        img2_show = np.zeros(img2.shape, np.uint8)
        img2_show = img2.copy()
        bar.next()
        for idx, point in enumerate(points2):
            pos = (point[0], point[1])
            cv2.putText(
                img2_show,
                str(idx),
                pos,
                fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                fontScale=0.1,
                color=(0, 255, 0),
            )
        bar.next()

        # print(2)
        # cv2.imwrite(Ui().save_file(num), img1_show) #
        # cv2.imwrite(Ui().save_file(num), img2_show) #
        # print("2 end")

        points = (1 - alpha) * np.array(points1) + alpha * np.array(points2)

        img1 = np.float32(img1)
        img2 = np.float32(img2)
        img_morphed = np.zeros(img1.shape, dtype=img1.dtype)

        triangles1 = get_triangles(points1)
        triangles2 = get_triangles(points2)
        triangles = get_triangles(points)
        bar.next()
        time.sleep(0.5)
        bar.finish()

        bar = IncrementalBar("Работа с треугольниками:", max=len(triangles) + 1)
        bar.next()
        for i in triangles:
            x = i[0]
            y = i[1]
            z = i[2]

            tri1 = [points1[x], points1[y], points1[z]]
            tri2 = [points2[x], points2[y], points2[z]]
            tri = [points[x], points[y], points[z]]

            rect1 = cv2.boundingRect(np.float32([tri1]))
            rect2 = cv2.boundingRect(np.float32([tri2]))
            rect = cv2.boundingRect(np.float32([tri]))

            tri_rect1 = []
            tri_rect2 = []
            tri_rect_warped = []

            for i in range(0, 3):
                tri_rect_warped.append(((tri[i][0] - rect[0]), (tri[i][1] - rect[1])))
                tri_rect1.append(((tri1[i][0] - rect1[0]), (tri1[i][1] - rect1[1])))
                tri_rect2.append(((tri2[i][0] - rect2[0]), (tri2[i][1] - rect2[1])))

            img1_rect = img1[rect1[1]: rect1[1] + rect1[3], rect1[0]: rect1[0] + rect1[2]]
            img2_rect = img2[rect2[1]: rect2[1] + rect2[3], rect2[0]: rect2[0] + rect2[2]]

            # cv2.imwrite(Ui().save_file(num), img1_rect) #
            # cv2.imwrite(Ui().save_file(num), img2_rect) #

            size = (rect[2], rect[3])
            warped_img1 = affine_transform(img1_rect, tri_rect1, tri_rect_warped, size)
            warped_img2 = affine_transform(img2_rect, tri_rect2, tri_rect_warped, size)

            img_rect = (1.0 - alpha) * warped_img1 + alpha * warped_img2

            mask = np.zeros((rect[3], rect[2], 3), dtype=np.float32)
            cv2.fillConvexPoly(mask, np.int32(tri_rect_warped), (1.0, 1.0, 1.0), 16, 0)

            img_morphed[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]] = (
                    img_morphed[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
                    * (1 - mask)
                    + img_rect * mask
            )
            bar.next()

        bar.finish()

        img_morphed = np.uint8(img_morphed)

        # Создаём копии изображений
        img_morphed_with_triangles = img_morphed.copy()
        img1_with_triangles = np.uint8(img1.copy())
        img2_with_triangles = np.uint8(img2.copy())

        # Отрисовываем треугольники
        draw_delaunay_triangles(img_morphed_with_triangles, points, triangles)
        draw_delaunay_triangles(img1_with_triangles, points1, triangles)
        draw_delaunay_triangles(img2_with_triangles, points2, triangles)

        print()
        print(f"Запись готового изображения ({num + 1})...")
        time.sleep(1)
        cv2.imwrite(Ui().save_file(num), img_morphed)

        cv2.imwrite(Ui().save_file(num), img_morphed_with_triangles)
        cv2.imwrite(Ui().save_file(num), img1_with_triangles)
        cv2.imwrite(Ui().save_file(num), img2_with_triangles)

        print(f"Done! ({num + 1})")
        print()
        time.sleep(1)
