import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import json
from flask import Flask,request
import librosa
import threading
import copy
from annoy import AnnoyIndex
from collections import Counter

from download_file import download_file 
from extract_features import toMFCC,crop_MFCC_100,toMFCC_100
from mapping import mappingTypeVietNam,mappingTypeAU,list_model
from Resnet18 import build_ResNet


import warnings
warnings.filterwarnings("ignore")


#------------------ LOAD MODEL ------------------#
print("--------Loading model")
model_40k_Res50_VN = load_model("Model/model_Res50_VN_7.h5")
model_40k_Res50_AU = load_model("Model/model_Res50_AU_15.h5")

model_Res18_VN = build_ResNet("ResNet18",9,13)
model_Res18_VN.load_weights("Model/model_Res18_VN_7.h5")

model_Res18_VN_2 = build_ResNet("ResNet18",9,20)
model_Res18_VN_2.load_weights("Model/model_Res18_VN_7_2.h5")

model_Res18_AU_2 = build_ResNet("ResNet18",11,20)
model_Res18_AU_2.load_weights("Model/model_Res18_AU_13_2.h5")

model_fìnd_similar = AnnoyIndex(100,metric='angular')
model_fìnd_similar.load('Model/model_find_similar.ann')

with open("Model/mapping_song.json", "r") as fp:
    mapping_song = json.load(fp)
print("--------Loading model OK")
#------------------ LOAD MODEL ------------------#

def predict_model(model_id, X):
    X = X[np.newaxis, ...]  # array shape (1, 130, 13, 1)
    if model_id == 0:
        return model_40k_Res50_VN.predict(X)[0]
    if model_id == 1:
        return model_40k_Res50_AU.predict(X)[0]
    if model_id == 2:
        return model_Res18_VN.predict(X)[0]
    if model_id == 3:
        return model_Res18_VN_2.predict(X)[0]
    if model_id == 4:
        return model_Res18_AU_2.predict(X)[0]

def find_similar_model(name_file):
    y, sr = librosa.load("Input/"+name_file, sr=16000)
    feat = toMFCC_100(y)
    results = []
    for i in range(0, feat.shape[0], 10):
        crop_feat = crop_MFCC_100(feat, i, nb_step=10)
        result = model_fìnd_similar.get_nns_by_vector(crop_feat, n=5)
        result_songs = [mapping_song['data'][k] for k in result]
        results.append(result_songs)
        
    results = np.array(results).flatten()

    print("result:")
    most_song = Counter(results)
    return most_song.most_common(5)


# initalize flask app
app = Flask(__name__)


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

@app.route('/')
def index():
    return {'msg':"pong"}

@app.route('/ping/', methods=['GET'])
def testAPI():
    return {'msg':"pong"}


@app.route('/predict_genre/', methods=['GET'])
def predict_genre():
    response = []
    name_file = request.args.get('name_file')
    print("Processing: "+name_file)
    type = request.args.get('type')
    model_id = int(request.args.get('model_id'))
    download_file(type,name_file)

    MFCCs = []
    MFCCs=toMFCC("Input"+"/"+name_file,num_mfcc = list_model[model_id]['n_mfcc'])
    
    _part = 0
    for mfcc in MFCCs:
        res_of_part = []
        if list_model[model_id]['country'] == 'VN':
            res_of_part = copy.deepcopy(mappingTypeVietNam)
        else:
            res_of_part = copy.deepcopy(mappingTypeAU)
        for segment in mfcc:
            rs_segment = predict_model(model_id, segment)
            for idx, val in enumerate(rs_segment):
                res_of_part[idx]['value'] += val*10
                #res[mappingTypeVietNam[idx]] += val*10

        obj = {'time':str(_part*30+15)+"--"+str(_part*30+45),'predict':res_of_part}
        _part+=1
        response.append(obj)

    #result all
    result_all = []
    if list_model[model_id]['country'] == 'VN':
        result_all = copy.deepcopy(mappingTypeVietNam)
    else:
        result_all = copy.deepcopy(mappingTypeAU)

    for part in response:
        for idx, res in enumerate(part['predict']):
            result_all[idx]['value'] += res['value']
    response.insert(0,{'time':'All','predict':result_all})
    return({'data':response})

@app.route('/find_similar/', methods=['GET'])
def find_similar():
    name_file = request.args.get('name_file')
    print("Processing: "+name_file)
    type = request.args.get('type')
    download_file(type,name_file)
    items = find_similar_model(name_file)
    res = []
    for item in items:
        res.append({"song_id":int(item[0]),"value":item[1]})
    return {'data':res}

def runLocaltunel():
    os.system('lt --port 8089 --subdomain bbbuit')
    print("Local tunel DIE or duplicate subdomain")
    
def runFlask():
    os.environ["WERKZEUG_RUN_MAIN"] = "true"
    port = int(os.environ.get('PORT', 8089))   
    app.run(host='0.0.0.0', port=port)

if __name__ == "__main__":
    # t1 = threading.Thread(target=runLocaltunel,args=())
    # t2 = threading.Thread(target=runFlask,args=())
    # jobs = []
    # jobs.append(t1)
    # jobs.append(t2)
    # # Start the threads (i.e. calculate the random number lists)
    # for j in jobs:
    #     j.start()

    # # Ensure all of the threads have finished
    # for j in jobs:
    #     j.join()

    runFlask()

