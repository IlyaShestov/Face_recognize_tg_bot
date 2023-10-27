import telebot
import User_dict
import recognize_face_on_photo
import Bot_attribute
import cv2
import os


TOKEN = Bot_attribute.bot_token  # заберем токен из файла  Bot_attribute.py

bot = telebot.TeleBot(TOKEN)  # объект класса бот

# распознаватель обученый на аугментированом сете
trained_recognizer_adv = cv2.face.LBPHFaceRecognizer_create()  # создаём экземпляр распознавателя
trained_recognizer_adv.read(
    'models/trained_recognizer_advanced.yml')  # загружаем модель обученую  на аугментированых данных

"""
Телеграмм бот попробует распознать человека на отправленом фото.
Если распознавание успешно, то вытащит данные из словаря с пользователями 
"""


@bot.message_handler(commands=['start'])  # приветствие
def send_welcome(message):
    bot.send_message(
        message.chat.id,
        'Привет!✌️\n' +
        'Я бот, попробую тебя узнать. \n' +
        'Пришли мне свое селфи. \n' +
        'Пожалуйста.\n'
    )


@bot.message_handler(content_types=['photo'])
def get_photo(message):
    file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)  # генерируем ссылку на фото из сообщения
    downloaded_file = bot.download_file(file_info.file_path)  # скаченый файл

    sourse = 'tg_download_files/' + file_info.file_path   # сформируем путь для сохранения
    with open(sourse, 'wb') as new_file:
        new_file.write(downloaded_file)  # запишем файл

    # пытаемся распознать лицо
    # если распознаем, то вытащим данные из словаря с известными пользователями
    predict_adv, user_name_adv, user_description_adv, conf_adv = recognize_face_on_photo.face_recognize(sourse,
                                                                                                        User_dict.user_dict,
                                                                                                        trained_recognizer_adv)
    # удаляем скаченый файл
    os.remove(sourse)

    # сформируем ответ в соответсвие с предсказанием
    if predict_adv is None:
        response = 'Похоже здесь никого нет '
    else:
        response = 'Да это же  {}!  {}  \n'.format(user_name_adv,
                                                   user_description_adv)

    # Вернем ответ бота
    bot.reply_to(message, response)


bot.infinity_polling()
