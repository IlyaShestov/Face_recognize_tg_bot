"""
Запуск файла дообучет модель распознавани лиц на вашет датасете
Забирает файл из папки  'data_set_advanced'
Сохраняет модель в папку models
"""

import cv2
import os
import numpy as np

data_path_adv = 'data_set_advanced'  # путь к датасету с аугментироваными фотографиями пользователей
recognizer_adv = cv2.face.LBPHFaceRecognizer_create()  # создаём распознаватель лиц


# создаем детектор лиц
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def get_images_and_labels(datapath):
    """
    Функция детектирует лица на датасете пользователя и размечает пары фото-лейбл  для дообучения модели
    :param datapath: путь к датасету с фото
    :return:  список лиц и соотвестующий лейблов(user_id)
    """

    train_paths = [os.path.join(datapath, f) for f in os.listdir(datapath)]  # получаем путь к трейн сету
    images = []
    labels = []
    counter = 0
    # перебираем все картинки
    for train_path in train_paths:

        image = cv2.imread(train_path)  # читаем картинку
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # переводим в ч/б

        # вытащим id из имени файла
        user_id = int(os.path.basename(train_path).split('.')[0].split('-')[1])
        # определяем лицо на картинке
        result_frame = detector.detectMultiScale(image)

        for (x, y, w, h) in result_frame:
            # добавляем его к списку картинок
            images.append(image[y: y + h, x: x + w])
            # добавляем id пользователя в список подписей
            labels.append(user_id)

        counter = counter + 1

    return images, labels, counter


if __name__ == "__main__":

    # учим модель на сете с аугментацией
    train_images_adv, train_labels_adv, count_adv = get_images_and_labels(data_path_adv)  # получаем список картинок  и лейблов
    recognizer_adv.train(train_images_adv, np.array(train_labels_adv))  # дообучаем модель на сформированном сете
    recognizer_adv.save('models/trained_recognizer_advanced.yml')   # сохраняем модель
    print('обработано фото: ', count_adv)
