
Идея: обучить модель компьютерного зрения что бы она могла распознать друзей по фото присланному в tg-bot
В качестве трейн-сета выступают фото из личного архива, видео с веб-камеры или короткое видео.
На фото детектируются лица, обрезаются и сохраняются для дальнейшей аугментации. Более подробно можно  посмотреть файле Preprocess_photo.ipynb
Обучение модели находится в файле  Train_model.py
Функция распознавания в файле recognize_face_on_photo.py
Файл с ботом в файле Bot_attribute.py

В процессе использованы библиотеки OpenCV и telebot. Зависимости в фале requirements.txt
Так же создан докер файл для развертывания в облаке

