FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt



ADD bot.py .
ADD recognize_face_on_photo.py .
ADD User_dict.py .
ADD Bot_attribute.py .



RUN ["mkdir", "/usr/src/app/tg_download_files"] 
RUN ["mkdir", "/usr/src/app/tg_download_files/photos"] 




WORKDIR /usr/src/app/models 

ADD haarcascade_frontalface_default.xml . 
ADD trained_recognizer_advanced.yml   .



WORKDIR /usr/src/app

CMD python ./bot.py