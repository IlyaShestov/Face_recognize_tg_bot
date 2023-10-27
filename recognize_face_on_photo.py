import cv2

# создаем детектор лиц
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def face_recognize(selfie, user_dict, trained_recognizer, conf_treshhold=50):
    """
    Функция детектирует и распознает лицо.
    Проверяет предсказания на наличие в словаре и выводит имя если есть совпадения
    :param selfie: фронтальное фото
    :param user_dict: словарь формата {id : [name, description]}
    :param trained_recognizer: обученый распознаватель cv2
    :param conf_treshhold: порог уверенности распознавателя
    :return: Вернет имя человека и описание, если не распозанано вернет predict = none b conf = none
             Если лицо распозанано, но значение уверенности(cof) выше tresholda вернет predict = 0,
    """
    frame = cv2.imread(selfie)  # читаем фото
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # переведем в ч/б для оптимизации и уменьшения каналов

    # обрабатываем кадр через нейросеть детекции лица
    detected_faces = detector.detectMultiScale(gray,
                                               scaleFactor=1.05,
                                               minNeighbors=15,
                                               minSize=(40, 40),
                                               flags=cv2.CASCADE_SCALE_IMAGE
                                               )

    predict = None  # если лицо не будет задетектировано значение останется None
    conf = None

    for (x, y, w, h) in detected_faces:
        # получим предсказание модели и уверенность
        predict, conf = trained_recognizer.predict(gray[y:y + h,
                                                   x:x + w])
        if conf > conf_treshhold:  # если не лицо не распознано вернет 0
            predict = 0

    user_name = user_dict[predict][0]  # вытаскиваем имя из словаря
    user_description = user_dict[predict][1]  # вытаскиваем описание из словаря

    return predict, user_name, user_description, conf
