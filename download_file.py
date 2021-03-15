import os
import requests

#HOST_BACKEND = "http://localhost:5000/api/v1/"
HOST_BACKEND = "http://103.92.29.98/api/v1/"
GET_REAL_FILE_BACKEND_API = HOST_BACKEND + "File/Musics/{}"
GET_TEMP_FILE_BACKEND_API = HOST_BACKEND + "File/Temp?nameFile={}&type={}"
SAVE_FILE_BACKEND_API = HOST_BACKEND + "File/Temp/SaveTempFile"

def download_file(type,name_file):
    if type == "real":
        url_file = GET_REAL_FILE_BACKEND_API.format(name_file)
        if not os.path.exists("Input/"+name_file):
            r = requests.get(url_file)
            open("Input/"+name_file,'wb').write(r.content)
    else:
        url_file = GET_TEMP_FILE_BACKEND_API.format(name_file,'audio')
        if not os.path.exists("Input/"+name_file):
            r = requests.get(url_file)
            open("Input/"+name_file,'wb').write(r.content)
