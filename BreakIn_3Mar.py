print('running 21Aug code')

import pandas as pd
import numpy as np
import glob
import json
import os
print('working directory = ', os.getcwd())
import sys
print('python version', sys.version)
import time
import datetime
import zipfile
import cv2
from os import path as op
import shutil
print('done till shutil')
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
print('Azure storage imported')
from azure.cosmosdb.table.tableservice import TableService
from azure.cosmosdb.table.models import Entity
from PIL import Image
print('image imported')
import tensorflow as tf

from object_detection.utils import label_map_util
#from sklearn.externals import joblib

#from azure.common import AzureMissingResourceHttpError
#from azure.storage.queue import QueueService

#from azure.storage.blob import BlockBlobService

 
#from azureml.core.model import Model

FINAL_TOP5_MODELS_REPLACE = ""
model_folder = ""
claim_no1 = ""
#rf_from_joblib =  ""
dentGraph =  ""
scratchGraph = ""
crackGraph =  ""
carGraph =  ""
meterGraph = ""
meterLRGraph = ""
partGraph = ""
npGraph = ""
odoGraph_1 = ""
odoGraph = ""
designLineGraph = ""
fpdamagesGraph = ""
model_folder = ""
path_label_template = ""
part_side_logic = ""
Replace_final = ""
metal_parts_with_D_ND = ""
pdtextdent = ""
pdtextcar = ""
pdtextcrack = ""
pdtextdesignLine = ""
pdtextfpdamages = ""
pdtextmeter = ""
pdtextmeterLR = ""
pdtextnp = ""
pdtextpart = ""
pdtextscratch = ""
pdtextRPM = ""
pdtextodo = ""
hitlim = 0.4
#City_tag_file = ""
#Repair_Data_all = "" 
master=""
sidecover_1=""
 

def loadAllModels():
    # dent,scratch,car, meter, crack, meterLR
    # Loads all the models in one go in different TfGraph objects
    # dent graph load
    dent_path_to_ckpt = op.join(
        model_folder, 'frozen_inference_graph_' + 'dent' + '.pb')

    dentGraph = tf.Graph()
    with dentGraph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(dent_path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

# scratch graph load
    scratch_path_to_ckpt = op.join(
        model_folder, 'frozen_inference_graph_' + 'scratch' + '.pb')

    scratchGraph = tf.Graph()
    with scratchGraph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(scratch_path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

# crack graph load
    crack_path_to_ckpt = op.join(
        model_folder, 'frozen_inference_graph_' + 'crack' + '.pb')

    crackGraph = tf.Graph()
    with crackGraph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(crack_path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


# car graph load
    car_path_to_ckpt = op.join(
        model_folder, 'frozen_inference_graph_' + 'car' + '.pb')

    carGraph = tf.Graph()
    with carGraph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(car_path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

# meter graph load
    meter_path_to_ckpt = op.join(
        model_folder, 'frozen_inference_graph_' + 'meter' + '.pb')
    meterGraph = tf.Graph()
    with meterGraph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(meter_path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


# meterLR graph load
    meterLR_path_to_ckpt = op.join(
        model_folder, 'frozen_inference_graph_' + 'meterLR' + '.pb')
    meterLRGraph = tf.Graph()
    with meterLRGraph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(meterLR_path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


    # part graph load
    part_path_to_ckpt = op.join(
        model_folder, 'frozen_inference_graph_' + 'part' + '.pb')

    partGraph = tf.Graph()
    with partGraph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(part_path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


# np graph load
    np_path_to_ckpt = op.join(
        model_folder, 'frozen_inference_graph_' + 'np' + '.pb')

    npGraph = tf.Graph()
    with npGraph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(np_path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

# # designline graph load
#     designLine_path_to_ckpt = op.join(
#         model_folder, 'frozen_inference_graph_' + 'designLine' + '.pb')

#     designLineGraph = tf.Graph()
#     with designLineGraph.as_default():
#         od_graph_def = tf.GraphDef()
#         with tf.gfile.GFile(designLine_path_to_ckpt, 'rb') as fid:
#             serialized_graph = fid.read()
#             od_graph_def.ParseFromString(serialized_graph)
#             tf.import_graph_def(od_graph_def, name='')

# fpdamages graph load
    fpdamages_path_to_ckpt = op.join(
        model_folder, 'frozen_inference_graph_' + 'fpdamages' + '.pb')

    fpdamagesGraph = tf.Graph()
    with fpdamagesGraph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(fpdamages_path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


# RPM graph load
    RPM_path_to_ckpt = op.join(
        model_folder, 'frozen_inference_graph_' + 'RPM' + '.pb')

    RPMGraph = tf.Graph()
    with RPMGraph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(RPM_path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
# Odo graph load
    odo_path_to_ckpt = op.join(
        model_folder, 'frozen_inference_graph_' + 'odo' + '.pb')

    odoGraph = tf.Graph()
    with odoGraph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(odo_path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')            

# Odo_1 graph load
    odo_1_path_to_ckpt = op.join(
        model_folder, 'frozen_inference_graph_' + 'odo_1' + '.pb')

    odoGraph_1 = tf.Graph()
    with odoGraph_1.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(odo_1_path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')   

    print("Loaded all models successfully")
    return(dentGraph, scratchGraph, crackGraph, carGraph, meterGraph, meterLRGraph, partGraph, fpdamagesGraph, npGraph, RPMGraph, odoGraph, odoGraph_1)

#dentGraph, scratchGraph, crackGraph, carGraph, meterGraph, meterLRGraph, partGraph, npGraph, designLineGraph, fpdamagesGraph = loadAllModels()


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def load_image_into_numpy_array(image):
    #Image.open(image).convert('RGB')
    #print('current wd ==', os.getcwd())
    (im_width, im_height) = image.size
    #print('image size is ===', image.size)
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)



#def unzip(zipFilePath, destDir):
 #   zfile = zipfile.ZipFile(zipFilePath)
 #   for name in zfile.namelist():
 #       (dirName, fileName) = os.path.split(name)
 #       # Check if the directory exisits
 #       newDir = destDir + '/' + dirName
 #       if not os.path.exists(newDir):
 #           os.mkdir(newDir)
 #       if not fileName == '':
 #           # file
 #           fd = open(destDir + '/' + name, 'wb')
 #           fd.write(zfile.read(name))
 #           fd.close()
 #   zfile.close()


def filter_images(local_folder_path):
    test_image_path = local_folder_path
    #print(test_image_path)
    test_imgs_all = glob.glob(test_image_path + "/*")
    # test_imgs = glob.glob(test_image_path + "/*")
    # print(test_imgs)
    print("test_imgs_all =============================",test_imgs_all)

    test_imgs_subset1 = [None] * len(test_imgs_all)
    for z in range(len(test_imgs_all)):
        if test_imgs_all[z].endswith((".jpg", ".jpeg", ".png")) and os.stat(test_imgs_all[z]).st_size > 0:
            test_imgs_subset1[z] = test_imgs_all[z]
    test_imgs_subset = [x for x in test_imgs_subset1 if x is not None]

    test_imgs = []
    test_imgs_temp = []
    if len(test_imgs_subset) > 0:
        test_imgs_temp = [None] * len(test_imgs_subset)
        for i in range(len(test_imgs_subset)):
            word = test_imgs_subset[i]
            if (word.find('SIDE') != -1):
                test_imgs_temp[i] = test_imgs_subset[i]
    test_imgs = [x for x in test_imgs_temp if x is not None]
    test_imgs_not_car = [i for i in test_imgs_subset if i not in test_imgs]
    return test_imgs_subset, test_imgs, test_imgs_not_car
    

###################INTERNAL for car and damage######################

def Flag_creation2(row):
    if row["bb0"] > row["bb0_x"] and row["bb2"] < row["bb2_x"] and row["bb1"] > row["bb1_x"] and row["bb3"] < row["bb3_x"]:
        return 1
    else:
        return 0


###################INTERNAL for part and damage######################


def Flag_creation4(row):
    if row["bb0_y"] > row["bb0_x"] and row["bb2_y"] < row["bb2_x"] and row["bb1_y"] > row["bb1_x"] and row["bb3_y"] < row["bb3_x"]:
        return 1
    else:
        return 0


##############INTERSECTION######################

def Flag_creation(row):
    if row["bb1_x"] < row["bb3_y"] and row["bb0_x"] < row["bb2_y"] and row["bb2_x"] > row["bb0_y"] and row["bb3_x"] > row["bb1_y"]:
        return 1
    else:
        return 0



def Flag_creation6_2(row):
    if row["Damage_Type"] == "Dent" and row["score_y"] < 0.45:
        return 1
    elif  row["Damage_Type"] == "Scratch" and row["score_y"] < 0.3:
        return 1
    elif  row["Damage_Type"] == "Bro_WS" and row["score_y"] < 0.7:
        return 1
    elif  row["Damage_Type"] == "Bro_HL" and row["score_y"] < 0.7:
        return 1
    elif  row["Damage_Type"] == "Bro_TL" and row["score_y"] < 0.7:
        return 1
    else:
        return 0

def Flag_creation15(row):
    if row["class"] == "Advocate_Symbol" and row["score"] > 0.85:
        return 1
    elif  row["class"] == "Press_Police" and row["score"] > 0.85:
        return 1
    elif  row["class"] == "PP_Flag" and row["score"] > 0.7:
        return 1
    elif  row["class"] == "PP_Symbol" and row["score"] > 0.7:
        return 1
    elif  row["class"] == "Police_Symbol" and row["score"] > 0.7:
        return 1
    else:
        return 0

#################ROW OPERATION ON DATA#############

def Flag_creation3(row):
    if row["class"] == row["Side"] :
        return 1
    else:
        return 0


######################################FP AND DL FUNCTIONS############################


def Flag_creation5(row):
    if row["bb0_y"] > row["bb0_FP"] and row["bb2_y"] < row["bb2_FP"] and row["bb1_y"] > row["bb1_FP"] and row["bb3_y"] < row["bb3_FP"]:
        return 1
    else:
        return 0    


def Flag_creation7(row):
    if row["bb0_y"] > row["bb0_dl"] and row["bb2_y"] < row["bb2_dl"] and row["bb1_y"] > row["bb1_dl"] and row["bb3_y"] < row["bb3_dl"]:
        return 1
    else:
        return 0
    
###################INTERACTION for part and damage######################

def Flag_creation8(row):
    if row["bb1_FP"] < row["bb3_y"] and row["bb0_FP"] < row["bb2_y"] and row["bb2_FP"] > row["bb0_y"] and row["bb3_FP"] > row["bb1_y"]:
        return 1
    else:
        return 0

def Flag_creation9(row):
    if row["bb1_dl"] < row["bb3_y"] and row["bb0_dl"] < row["bb2_y"] and row["bb2_dl"] > row["bb0_y"] and row["bb3_dl"] > row["bb1_y"]:
        return 1
    else:
        return 0    

    



def Flag_creation6(row):
    if row["damage_dist_FP"] > 60 and row["damage_dist_FP"] < 100:
        return 1
    else:
        return 0
    
    
 
def Flag_creation10(row):
    if row["damage_dist_dl"] > 80 and row["damage_dist_dl"] < 100:
        return 1
    else:
        return 0   


def Flag_creation12(row):
    if row["Area_FP"] == 2:
        return 1
    else:
        return 0
    
def Flag_creation13(row):
    if row["Area_dl"] == 2:
        return 1
    else:
        return 0  


def Flag_creation50(row):
    if row["damage_dist"] >=  35:
        return 1
    else:        
        return 0

def Flag_creation100(row):
    if row["bb1_x"] < row["bb3"] and row["bb0_x"] < row["bb2"] and row["bb2_x"] > row["bb0"] and row["bb3_x"] > row["bb1"]:
        return 1
    else:
        return 0


def check_blur(test_imgs):
    print("-----Checking blur-----")
    NBlur = [None] * len(test_imgs)
    Blur = [None] * len(test_imgs)
    flag = [None] * len(test_imgs)
    for i in range(len(test_imgs)):
        image = cv2.imread(test_imgs[i])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)
        if fm < 23:
            Blur[i] = test_imgs[i]
            flag[i] = "Image is blur"
        else:
            NBlur[i] = test_imgs[i]

    NBlur1 = [x for x in NBlur if x is not None]
    Blur1 = [x for x in Blur if x is not None]

    #print("Blur1", Blur1)
    if len(Blur1) != 0:
        Output_Blur = pd.DataFrame(np.column_stack((Blur, flag)))
        Output_Blur = Output_Blur.rename(index=str, columns={
                                         Output_Blur.columns[0]: "image", Output_Blur.columns[1]: "Damage_Type"})
        Output_Blur = Output_Blur.reset_index(drop=True)
        Output_Blur = Output_Blur.loc[Output_Blur['Damage_Type'] == "Image is blur"]
        Output_Blur['id'] = Output_Blur["image"].str.split(
            claim_no1, n=0, expand=True)[1]
        
        #print('out_put_bllluuuurrrrrr---------', Output_Blur)
        
        #print("Output_Blur", Output_Blur[['id', 'Damage_Type']])
        Output_Blur1 = Output_Blur[['id', 'Damage_Type']]
        Output_Blur1["score_y"] = ""
        Output_Blur1["class_x"] = ""
        Output_Blur1["score_x"] = ""
        Output_Blur1["Side"] = ""
        Output_Blur1["score"] = ""
        Output_Blur1["part_damage_dist"] = ""
    else:
        
        Output_Blur1 = pd.DataFrame(columns = ['id', 'Damage_Type', "score_y", "class_x", "score_x", "Side", "score", "part_damage_dist"], dtype=object)
    
    #print("OUTPUT_Blur1",  Output_Blur1)
    #Output_Blur1.to_csv('output_blur.csv', index = False)
    #print("NBlur1",  NBlur1 )
    #print("Blur1",  Blur1 )
    
    if len(Output_Blur1) == 0:
        Output_Blur1 = Output_Blur1
    else:
        Output_Blur1 = final_blur_output(Output_Blur1)
    
    return(NBlur1, Output_Blur1, Blur1)


def tf_od_pred(model_n, NBlur1):
    # path_to_ckpt = op.join(
    #     model_folder, 'frozen_inference_graph_' + model_n + '.pb')
    #path_to_label = op.join(os.getcwd(), str.replace(path_label_template, "modelName", model_n))
    #path_to_label = op.join(model_folder, str.replace(path_label_template, "modelName", model_n))
    if (model_n == 'car'):
        label_map=pdtextcar
        num_classes = 90
    elif (model_n == 'meter'):
        label_map=pdtextmeter
        num_classes = 4
    elif (model_n == 'crack'):
        label_map=pdtextcrack
        num_classes = 6
    elif (model_n == 'meterLR'):
        label_map=pdtextmeterLR
        num_classes = 8
    elif (model_n == 'part'):
        label_map=pdtextpart
        num_classes = 25
    elif (model_n == 'np'):
        label_map=pdtextnp
        num_classes = 6    
    # elif (model_n == 'designLine'):
    #     label_map=pdtextdesignLine
    #     num_classes = 18
    elif (model_n == 'fpdamages'):
        label_map=pdtextfpdamages
        num_classes = 15
    elif (model_n == 'dent'):
        label_map=pdtextdent
        num_classes = 3
    elif (model_n == 'scratch'):
        label_map=pdtextscratch
        num_classes = 8
    elif (model_n == 'RPM'):
        label_map=pdtextRPM
        num_classes = 4
    elif (model_n == 'odo'):
        label_map=pdtextodo
        num_classes = 5   
    elif (model_n == 'odo_1'):
        label_map=pdtextodo
        num_classes = 5   
    else:
        num_classes = 1
        

    if ((model_n == 'dent')):
        hitlim = 0.45
    elif ((model_n == 'scratch')):
        hitlim = 0.3
    elif ((model_n == 'np')):
        hitlim = 0.9
    elif ((model_n == 'odo')):
        hitlim = 0.1
    elif ((model_n == 'part')):
        hitlim = 0.3
    else:
        hitlim = 0.7


    data_result = pd.DataFrame(columns=['image', 'class', 'score', 'bb0', 'bb1', 'bb2', 'bb3'], dtype=object)
    #label_map = label_map_util.load_labelmap(path_to_label)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    detection_config = tf.ConfigProto()
    # detection_config.gpu_options.per_process_gpu_memory_fraction = 0.2
    # detection_config.gpu_options.per_process_gpu_memory_quota_mb = gpu_memory_limit
    detection_config.gpu_options.allow_growth = True

    # detection_graph = tf.Graph()
    # with detection_graph.as_default():
    #     od_graph_def = tf.GraphDef()
    #     with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
    #         serialized_graph = fid.read()
    #         od_graph_def.ParseFromString(serialized_graph)
    #         tf.import_graph_def(od_graph_def, name='')
    if(model_n == 'dent'):
        detection_graph = dentGraph
    elif(model_n == 'scratch'):
        detection_graph = scratchGraph
    elif(model_n == 'crack'):
        detection_graph = crackGraph    
    elif(model_n == 'car'):
        detection_graph = carGraph
    elif(model_n == 'meterLR'):
        detection_graph = meterLRGraph
    elif(model_n == 'part'):
        detection_graph = partGraph
    elif(model_n == 'np'):
        detection_graph = npGraph
    # elif(model_n == 'designLine'):
    #     detection_graph = designLineGraph
    elif(model_n == 'fpdamages'):
        detection_graph = fpdamagesGraph
    elif(model_n == 'RPM'):
        detection_graph = RPMGraph
    elif(model_n == 'odo'):
        detection_graph = odoGraph
    elif(model_n == 'odo_1'):
        detection_graph = odoGraph_1
    else:
        detection_graph = meterGraph

    with detection_graph.as_default():
        with tf.Session(config=detection_config, graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name(
                'detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name(
                'detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name(
                'detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            for image_path in NBlur1:
                image = Image.open(image_path)
                image_np = load_image_into_numpy_array(image)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores,
                        detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Write the results to hitlist - one line per hit over the 0.5
                nprehit = scores.shape[1]  # 2nd array dimension
                for j in range(nprehit):
                    fname = image_path
                    classid = int(classes[0][j])
                    classname = category_index[classid]["name"]
                    score = scores[0][j]
                    if (score >= hitlim):
                        sscore = float(str(score))
                        bbox = boxes[0][j]
                        b0 = float(str(bbox[0]))
                        b1 = float(str(bbox[1]))
                        b2 = float(str(bbox[2]))
                        b3 = float(str(bbox[3]))
                        df = pd.DataFrame(columns=['image', 'class', 'score', 'bb0', 'bb1', 'bb2', 'bb3'], dtype=object)
                        df.loc[0] = [fname, classname, sscore, b0, b1, b2, b3]
                        data_result = data_result.append(df, ignore_index=True)
                        #print(data_result)
    return(data_result)


def Damage_Intersection(data):
    #print("data:" , data)
    #data.to_csv("data_c.csv", index = False)
    data['C'] = data.reset_index().index
    data2 = data
    data3 = pd.merge(data, data2, left_on='image', right_on='image', how = 'outer')
    data3 = data3.assign(Flag=data3.apply(Flag_creation, axis=1))
    data3 = data3[data3['C_x'] != data3['C_y']]
    data3 = data3[data3["Flag"] == 1]
    data3["bb0_z"] = data3[["bb0_x", "bb0_y"]].max(axis=1)
    data3["bb1_z"] = data3[["bb1_x", "bb1_y"]].max(axis=1)
    data3["bb2_z"] = data3[["bb2_x", "bb2_y"]].min(axis=1)
    data3["bb3_z"] = data3[["bb3_x", "bb3_y"]].min(axis=1)
    data3["Internal_Area"] = (data3["bb2_z"] - data3["bb0_z"]) * (data3["bb3_z"] - data3["bb1_z"])
    data3["Area1"] = (data3["bb2_x"] - data3["bb0_x"]) * (data3["bb3_x"] - data3["bb1_x"])
    data3["Area2"] = (data3["bb2_y"] - data3["bb0_y"]) * (data3["bb3_y"] - data3["bb1_y"])
    data3["Area3"] = data3[['Area1','Area2']].min(axis=1)
    data3["AreaCover"] = data3['Internal_Area']*100/data3['Area3']

    #data4 = data3[data3['AreaCover'] >= 40]
    data4 = data3
    data4['Dedup'] = np.where(data4['C_y']> data4['C_x'], 'Original', 'Duplicate')
    data4 = data4[data4['Dedup'] == 'Original']
    data4 = data4[data4.class_x == data4.class_y]
    #data4.to_csv('data4_1.csv')


    if len(data4) > 0 and len(data) > 1 :
        data4['Row'] = data4.reset_index().index
        d1 = data4[['C_x', 'Row']]
        d2 = data4[['C_y', 'Row']]
        d2 = d2.rename(index=str, columns={d2.columns[0]: "C_x"})
        d3 = d2.append(d1, ignore_index=True)
        C_x = pd.DataFrame(np.unique(d3['C_x']))
        d3_min = d3.groupby(['C_x'])['Row'].transform(min) == d3['Row']
        d_sub = d3[d3_min]
        d4 = pd.merge(d3, d_sub[['C_x', 'Row']], on='C_x', how='left')
        d5 = d4[['Row_x', 'Row_y']]
        d6 = d5.drop_duplicates(subset=['Row_x', 'Row_y'], inplace=False)
        d6 = d6.sort_values("Row_y", inplace = False)
        d7 = d6.drop_duplicates(subset=['Row_x'], inplace=False)
        d8 = pd.merge(d3, d7[['Row_x', 'Row_y']],how = 'inner', left_on = "Row", right_on =  "Row_x")
        d9 = pd.merge(d8, data, how = 'inner', left_on = "C_x", right_on = "C")
        bb0 = pd.DataFrame((d9.groupby(['Row_y'])['bb0'].min()))
        bb0['index1'] = bb0.index 
        bb1 = pd.DataFrame((d9.groupby(['Row_y'])['bb1'].min()))
        bb1['index1'] = bb1.index
        bb2 = pd.DataFrame((d9.groupby(['Row_y'])['bb2'].max()))
        bb2['index1'] = bb2.index
        bb3 = pd.DataFrame((d9.groupby(['Row_y'])['bb3'].max()))
        bb3['index1'] = bb3.index
        score = pd.DataFrame((d9.groupby(['Row_y'])['score'].max()))
        score['index1'] = score.index
        image1 = d9[['Row_y', 'image']]
        image2 = image1.drop_duplicates(subset=['Row_y', 'image'], inplace=False) 
        image2 = image2.rename(index=str, columns={image2.columns[0]: "index1"})
        class1 = d9[['Row_y', 'class']]
        class2 = class1.drop_duplicates(subset=['Row_y', 'class'], inplace=False)
        class2 = class2.rename(index=str, columns={class2.columns[0]: "index1"})
        C1 = pd.DataFrame((d9.groupby(['Row_y'])['C_x'].max()))
        C1['index1'] = C1.index
        data5 = pd.DataFrame(np.column_stack((bb0[['index1', 'bb0']], 
                                            bb1['bb1'], bb2['bb2'], 
                                            bb3['bb3'], score['score'],
                                            image2['image'], class2['class'], C1['C_x'])))
        data5 = data5.rename(columns = {0 : 'index_o',  1 : 'bb0', 2 : "bb1", 3 : "bb2", 4 : "bb3", 5 : "score", 6 : "image", 7: "class", 8: "C"})
        data6 = data5[['image', 'class', 'score', 'bb0', 'bb1', 'bb2', 'bb3', 'C']]
        data7 = data[(data['C'].isin(C_x[0]) == False)]
        data9 = data7.append(data6, ignore_index=True)
    else :
        data9 = data
    udamages = data9.drop('C', 1)
    #print("udamages:" , udamages)
    return(udamages)

def damage_part(data_p, data_d):
    #print('------data_p-------', data_p)
    
    data_p['C'] =  data_p.reset_index().index
    data_p['id'] = data_p["image"].str.split(claim_no1, n=0, expand=True)[1]
    #data_p.to_csv("data_p.csv", index = False)
    data_p['id1']= data_p['id'].astype(str) + data_p['class'].astype(str)
    #data_p['C'] = data_p.reset_index().index
    data_p_sub = data_p[data_p['class'].isin(["SIDEDOOR", "HEADLAMP", "TAILLAMP"])]
    #data_p_sub.to_csv("data_p_sub.csv", index = False)

    data_p_sub2 = data_p[~data_p['class'].isin(["SIDEDOOR", "HEADLAMP", "TAILLAMP"])]
    idp = data_p_sub2.groupby(['image', 'class'])['score'].transform(max) == data_p_sub2['score']
    data_p_sub3 = data_p_sub2[idp]
    data_p_sub3 = data_p_sub3[data_p_sub3['class'] != 'CAR']
    #data_p_sub3.to_csv("data_p_sub3.csv", index = False)
    
    data_p_dup = data_p_sub
    data_p_merge1 = pd.merge(data_p_sub, data_p_dup, left_on='id1', right_on='id1', how = 'outer')
    data_p_merge2 = data_p_merge1[data_p_merge1['C_x'] != data_p_merge1['C_y']]
    #data_p_merge1.to_csv("data_p_merge1.csv", index = False)
    #data_p_merge2.to_csv("data_p_merge2.csv", index = False)
    data_p_merge2['Dedup'] = np.where(data_p_merge2['C_y']> data_p_merge2['C_x'], 'Original', 'Duplicate')
    data_p_merge2 = data_p_merge2[data_p_merge2['Dedup'] == 'Original']
    
    if len(data_p_merge2) > 0:
        data_p_merge2 = data_p_merge2.assign(Flag_n=data_p_merge2.apply(Flag_creation, axis=1))
        #data_p_merge2.to_csv("data_p_merge2.csv", index = False)
        #data_p_merge3 = data_p_merge2[data_p_merge2['Flag_n'] == 1]
        data_p_merge3 = data_p_merge2
        data_p_merge3["bb0_z"] = data_p_merge3[["bb0_x", "bb0_y"]].max(axis=1)
        data_p_merge3["bb1_z"] = data_p_merge3[["bb1_x", "bb1_y"]].max(axis=1)
        data_p_merge3["bb2_z"] = data_p_merge3[["bb2_x", "bb2_y"]].min(axis=1)
        data_p_merge3["bb3_z"] = data_p_merge3[["bb3_x", "bb3_y"]].min(axis=1)
        data_p_merge3["Internal_Area"] = (data_p_merge3["bb2_z"] - data_p_merge3["bb0_z"]) * (data_p_merge3["bb3_z"] - data_p_merge3["bb1_z"])
        data_p_merge3["Area1"] = (data_p_merge3["bb2_x"] - data_p_merge3["bb0_x"]) * (data_p_merge3["bb3_x"] - data_p_merge3["bb1_x"])
        data_p_merge3["Area2"] = (data_p_merge3["bb2_y"] - data_p_merge3["bb0_y"]) * (data_p_merge3["bb3_y"] - data_p_merge3["bb1_y"])
        data_p_merge3["Area3"] = data_p_merge3[['Area1','Area2']].min(axis=1)
        data_p_merge3["AreaCover"] = data_p_merge3['Internal_Area']*100/data_p_merge3['Area3']
        data_p_merge3["AreaCover1"] = data_p_merge3["AreaCover"]*data_p_merge3["Flag_n"]
        #data_p_merge3.to_csv("data_p_merge3.csv", index = False)
        data_p_merge4 = data_p_merge3[data_p_merge3['AreaCover'] <= 20]
        data_p_merge4["Criteria1"] =  data_p_merge4["score_x"]*data_p_merge4["Area1"]
        data_p_merge4["Criteria2"] =  data_p_merge4["score_y"]*data_p_merge4["Area2"]
        #data_p_merge4.to_csv("data_p_merge4.csv", index = False)
        data_p_merge4["rank1"] = data_p_merge4.groupby(["id_x", "class_x"])["Criteria1"].rank(ascending=False)
        data_p_merge4["rank2"] = data_p_merge4.groupby(["id_y", "class_y"])["Criteria2"].rank(ascending=False)
        data_p_merge4["rank"] = data_p_merge4["rank2"] + data_p_merge4["rank1"]
        #data_p_merge4.to_csv("data_p_merge4.csv", index = False)
        idz = data_p_merge4.groupby(['id_x', 'class_x'])['rank'].transform(max) == data_p_merge4['rank']
        data_p_merge5 = data_p_merge4[idz]
        #data_p_merge5.to_csv("data_p_merge5.csv", index = False)

        l1 = data_p_merge5.C_x.unique()
        l2 = data_p_merge5.C_y.unique()
        #l2 = np.unique[data_p_merge5['C_y']]
        l3 = np.concatenate([l1, l2])
        data_p_sub4 = data_p[data_p['C'].isin(l3)]
        #data_p_sub4.to_csv("data_p_sub4.csv", index = False)
        data_p_subset = data_p_sub4.append(data_p_sub3, ignore_index=True)
    else :
        data_p_subset = data_p_sub3

    #idp = data_p.groupby(['image', 'class'])['score'].transform(max) == data_p['score']
    #data_p_subset = data_p[idp]
    #data_p_subset = data_p_subset[data_p_subset['class'] != 'CAR']
    data_p1 = pd.merge(data_p_subset, data_d, left_on='id', right_on='id', how = 'right')
    data_p1 = data_p1.assign(Flag=data_p1.apply(Flag_creation, axis=1))
    #data_p1.to_csv("data_p1.csv", index = False)
    #data_p2 = data_p1[data_p1['Flag'] == 1]
    data_p2 = data_p1
    data_p2["bb0_z"] = data_p2[["bb0_x", "bb0_y"]].max(axis=1)
    data_p2["bb1_z"] = data_p2[["bb1_x", "bb1_y"]].max(axis=1)
    data_p2["bb2_z"] = data_p2[["bb2_x", "bb2_y"]].min(axis=1)
    data_p2["bb3_z"] = data_p2[["bb3_x", "bb3_y"]].min(axis=1)
    data_p2["inter_area"] = (data_p2["bb2_z"] - data_p2["bb0_z"]) * (data_p2["bb3_z"] - data_p2["bb1_z"])
    data_p2["damage_area"] = (data_p2["bb2_y"] - data_p2["bb0_y"]) * (data_p2["bb3_y"] - data_p2["bb1_y"])
    data_p2["part_area"] = (data_p2["bb2_x"] - data_p2["bb0_x"]) * (data_p2["bb3_x"] - data_p2["bb1_x"])
    data_p2["Flag_in"] = np.where(data_p2['inter_area'] > data_p2["damage_area"], 1, 0)
    data_p2["damage_dist"] = 100*data_p2["inter_area"]/data_p2["damage_area"]
    data_p2["part_damage_dist"] = 100*data_p2["inter_area"]/data_p2["part_area"]
    #data_p2.to_csv("data_p2.csv", index = False)
    #print("part_damage:" , data_p2)
    
    data_p2 = data_p2.assign(Flag_filter1=data_p2.apply(Flag_creation50, axis=1))
    data_p2 = data_p2[data_p2['Flag_filter1'] == 1]
    data_p2 = data_p2.drop(['Flag_filter1'], axis=1)
    
    return(data_p2)


def damage_part_meterLR(data_LR, data_d):
    data_LR['C_s'] =  data_LR.reset_index().index
    data_LR['id'] = data_LR["image"].str.split(claim_no1, n=0, expand=True)[1]
    idsi = data_LR.groupby(['image'])['score'].transform(max) == data_LR['score']
    data_LR_subset = data_LR[idsi]
    data_LR1 = pd.merge(data_d, data_LR_subset, left_on='id', right_on='id', how = 'inner')
    #print("part_damage_LR:" , data_LR1)
    #data_LR1.to_csv("data_LR_1.csv", index = False)
    return(data_LR1)




def prepare_output(car_final, Output, data_dent, data_scratch, data_crack, data_FP, car_not_found, Blur1, data_angle, data_part): # , data_np):
    print('=============Preparing Final Output==============')
    #print('===========Output========', Output)
    len_not_car = 0
    l_angle_final = 0
    car_angle = 0
    car_final = car_final[['id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x', 'Area']]
    l_car_final = len(np.unique(car_final['image']))
    #Output = pd.DataFrame(columns=['id', 'Damage_Type'])
    #car_final, l_car_final, Output = merge_car_meter(data_car, data_meter, Output)
    #print('=========car_final======', car_final)
    #print("********l_car_final***************", l_car_final)
    #print('data angle', data_angle)
     
    
    if data_angle is None:
        data_angle = pd.DataFrame(columns= ['image', 'class', 'score', 'bb0', 'bb1', 'bb2', 'bb3'], dtype=object)
    
    if l_car_final == 0:
        Output.loc[0] = ['Summary', 'No Car Image Found']
        
        #Output_np = pd.DataFrame(columns= ['id', 'Damage_Type', 'class_x', 'score_x', 'Side', 'score', 'part_damage_dist'])
        
    elif l_car_final > 0 and l_car_final <= 7:
        #l_angle_final, car_angle = merge_car_angle(car_final, data_angle)
        l_angle_final, car_angle,status_180,proper_angle_image = merge_car_angle_1(car_final, data_angle,data_part,master,sidecover_1, Breakin_id)
        #car_angle.to_csv('car_angle.csv', index =False)
        #np_list, Output_np = merge_car_np(car_final, data_np)
        
        #car_angle.to_csv('car_angle.csv', index =False)
        print("180 availability =" ,status_180) 
        if(proper_angle_image == 1 and status_180 == 1):
            print("inside 180 check and proper angle check...........")
        
            len_not_car = 0
            
            not_car = car_not_found
            #print('-----Car not found-----', not_car)
            not_car = list(not_car)
            not_car_label = [None] * len(not_car)
            len_not_car = len(not_car)
            #print("not_car_length...", len(not_car))
            if len_not_car != 0:
                not_car = list(not_car)
                not_car_label = [None] * len(not_car)
                for j in range(len(not_car)):
                    not_car_label[j] = "Car not found"
                Output_car = pd.DataFrame(
                    np.column_stack((not_car, not_car_label)))
                Output_car = Output_car.rename(index=str, columns={
                                               Output_car.columns[0]: "image", Output_car.columns[1]: "Damage_Type"})
                
        
                #print('=====output_car======', Output_car)        
                Output_car = Output_car.rename(columns={"image": "id"})
                Output_car = Output_car[['id', 'Damage_Type']]
                Output = Output.append(Output_car)
                #print("Output_temp : ", Output)
                Output_car1 = Output
                Output_car1["score_y"] = ""
                Output_car1["class_x"] = ""
                Output_car1["score_x"] = ""
                Output_car1["Side"] = ""
                Output_car1["score"] = ""
                Output_car1["part_damage_dist"] = ""
                Output = Output_car
            #print("Car_output******", Output)
        else:
            print("180 check and proper angle check failed ...........")
            # Output = pd.DataFrame(columns = ['id','Damage_Type','score_y','class_x','score_x','Side	score','part_damage_dist'])
            Output.loc[0] = ['Summary', 'Less than 8 images have car/few images are blur']
            
            car_angle = pd.DataFrame(columns = ['image','class','score','bb0','bb1','bb2','bb3','id'], dtype=object)
            
            # Output_np = pd.DataFrame(columns= ['id', 'Damage_Type', 'class_x', 'score_x', 'Side', 'score', 'part_damage_dist']) 
    
    else:
        #np_list, Output_np = merge_car_np(car_final, data_np)
        # l_angle_final, car_angle = merge_car_angle(car_final, data_angle)
        l_angle_final, car_angle,status_180,proper_angle_image = merge_car_angle_1(car_final, data_angle,data_part,master,sidecover_1,Breakin_id)
        print("180 availability =" ,status_180) 
        if(proper_angle_image == 1 and status_180 == 1):
        #car_angle.to_csv('car_angle.csv', index =False)
        #print("car_angle", car_angle)
        #print('car_angle--------', car_angle)
        #print("car_angle list of variables: ", list(car_angle))
        #print('==========l_angle_final======', l_angle_final)
        #l_angle_final = l_angle_final[0]
            if l_angle_final >= 4:
                #print('l_angle_final is greater than or equal to 4')
                dent_list, Output, dent_part = merge_car_dent(car_final, data_dent, Output, data_part)
                #print("dent_output******", Output)
                scratch_list, Output, scratch_part = merge_car_scratch(car_final, data_scratch, Output, data_part)
                #print("scratch_output******", Output)
                crack_list, Output, car_crack3 = merge_car_crack(car_final, data_crack, Output)
                Subset_Damage = Output[Output['Damage_Type'].isin(["Scratch", "Dent", "Crack"])]
                #print("Subset_Damage:" , Subset_Damage)
                #print("dent_part:" , dent_part)
                #print("scratch_part:" , scratch_part)
                #print("car_crack3:" , car_crack3)
                if dent_part is None and scratch_part is None:
                    damage_part = None
                elif scratch_part is None and dent_part is not None:
                    damage_part = dent_part
                elif scratch_part is not None and dent_part is None:
                    damage_part = scratch_part
                else:
                    damage_part = dent_part.append(scratch_part)
                #print("car_final :" , car_final)
                carlist = np.unique(car_final["id"])
                #damage_part.to_csv("damage_part.csv")
                if (damage_part) is not None:
                    damage_part = damage_part[damage_part['id'].isin(carlist)]
                    #print("damage_part" , damage_part)
                    l2 = len(np.unique(damage_part['id']))
                    #print("l2 (((((((((((((((((((((((((((((((:", l2)
                else :
                    l2 = 0
                #print("l2 (((((((((((((((((((((((((((((((:", l2)
                if l2 == 0 and (car_crack3) is None:
                    Output = pd.DataFrame(columns=['id', 'Damage_Type'], dtype=object)
                    Output.loc[len(Output)+1] = ['Dent', 'Not Found']
                    Output.loc[len(Output)+1] = ['Scratch', 'Not Found']
                    Output.loc[len(Output)+1] = ['Crack', 'Not Found']
                    #print("Output_without_damage : ", Output)
                elif l2 == 0 and (car_crack3) is not None:
                    Output = car_crack3[['id', 'class', 'score']]
                    Output = Output.rename(columns = {'class': 'Damage_Type'})
                    Output = Output.rename(columns = {'score': 'score_y'})
                elif l2 != 0:
                    #Output_temp = damage_part
                    #print("damage_part : " ,damage_part)
                    damage_part_side = None
                    damage_part_side = damage_part_meterLR(data_angle, damage_part)
                    #damage_part_side.to_csv("check.csv", index=False)
                    #print("list of variables", list(damage_part_side))
                    dent_scratch_f = dent_scratch_final(damage_part_side, data_FP)
                    if len(dent_scratch_f) > 0:
                        if (car_crack3) is None:
                            dent_scratch_crack_f = dent_scratch_f
                        else :
                            car_crack4 = car_crack3[['id', 'class', 'score']]
                            car_crack4 = car_crack4.rename(columns = {'class': 'class_y'})
                            car_crack4 = car_crack4.rename(columns = {'score': 'score_y'})
                            car_crack4["class_x"] = ""
                            car_crack4["score_x"] = ""
                            car_crack4["Side"] = ""
                            car_crack4["score"] = ""
                            car_crack4["part_damage_dist"] = ""
                            #print("car_crack_4 :", car_crack4)
                            dent_scratch_crack_f = dent_scratch_f.append(car_crack4)
                            #print("dent_scratch_crack_f :", dent_scratch_crack_f)
                        Output =  dent_scratch_crack_f
                        Output = Output.rename(columns = {'class_y': 'Damage_Type',})
                        #print("Output issssssss:", Output)
                    else :
                        if  (car_crack3) is None:
                            Output = pd.DataFrame(columns=['id', 'Damage_Type'], dtype=object)
                            Output.loc[len(Output)+1] = ['Dent', 'Not Found']
                            Output.loc[len(Output)+1] = ['Scratch', 'Not Found']
                            Output.loc[len(Output)+1] = ['Crack', 'Not Found']
                        else :
                            Output = car_crack3[['id', 'class', 'score']]
                            Output = Output.rename(columns = {'class': 'Damage_Type'})
                            Output = Output.rename(columns = {'score': 'score_y'})
                else :
                    Output = pd.DataFrame(columns=['id', 'Damage_Type'], dtype=object)
                    Output.loc[len(Output)+1] = ['Dent', 'Not Found']
                    Output.loc[len(Output)+1] = ['Scratch', 'Not Found']
                    Output.loc[len(Output)+1] = ['Crack', 'Not Found']
            else:
                Output = pd.DataFrame(columns=['id', 'Damage_Type'], dtype=object)
                Output = Output.reset_index(drop=True)
                Output.loc[len(Output)+1] = ['Car Images', 'do not have min no. of angles']
                #print("Output :" , Output)
                #print("list of variables", list(Output)) 
        else:
            print("180 check and proper angle check failed ...........")
            # Output = pd.DataFrame(columns = ['id','Damage_Type','score_y','class_x','score_x','Side	score','part_damage_dist'])
            Output.loc[0] = ['Summary', 'Less than 8 images have car/few images are blur']
            
            car_angle = pd.DataFrame(columns = ['image','class','score','bb0','bb1','bb2','bb3','id'], dtype=object)
            
            # Output_np = pd.DataFrame(columns= ['id', 'Damage_Type', 'class_x', 'score_x', 'Side', 'score', 'part_damage_dist'])   
            
    if len(Output.columns) == 2:
        Output["score_y"] = ""
        Output["class_x"] = ""
        Output["score_x"] = ""
        Output["Side"] = ""
        Output["score"] = ""
        Output["part_damage_dist"] = ""
    elif len(Output.columns) == 3:
        Output["class_x"] = ""
        Output["score_x"] = ""
        Output["Side"] = ""
        Output["score"] = ""
        Output["part_damage_dist"] = ""
    else :
        Output = Output
    Output = Output[["id", 'Damage_Type', "score_y", "class_x", "score_x", "Side","score", "part_damage_dist"]]
    if len_not_car != 0:
        Output = Output.append(Output_car1)
    
    l_angle_final = l_angle_final 
    #Output.to_csv("Output_check_today.csv", index = False)
    return (Output, l_angle_final, car_angle)
    #Output.to_csv("Output_check_today.csv", index = False)
    #return (Output, l_angle_final, car_angle)



def merge_car_meter(data_car, data_meter, Output):
    print("data_car-----------------------------",data_car)
    print("data_meter-----------------------------",data_meter)
    if (len(data_car) != 0 and len(data_meter) != 0):
        print("inside the if of merge_car_meter--------------------------")
        data_car['id'] = data_car["image"].str.split(
            claim_no1, n=0, expand=True)[1]
        data_car["Area"] = (data_car["bb2"] - data_car["bb0"]) *(data_car["bb3"] - data_car["bb1"])
        data_car["Criteria"] = data_car["score"] * data_car["Area"]
        idx = data_car.groupby(['image'])['Criteria'].transform(max) == data_car['Criteria']
        car_data_subset = data_car[idx]
        car_data_subset2 = car_data_subset[car_data_subset['class'] == 'car']
        #l_car_final = 0
        car_final = None
        l_c = len(np.unique(car_data_subset2['image']))
        if (l_c == 0):
            Output = Output.reset_index(drop=True)
            Output.loc[0] = ['Car', 'Not Found']
        else:
            l_m = len(np.unique(data_meter['image']))
            if l_m == 0:
                Output = Output.reset_index(drop=True)
                Output.loc[0] = ['Car', 'Not Found']
            else:
                data_meter["Area"] = (
                    data_meter["bb2"] - data_meter["bb0"]) * (data_meter["bb3"] - data_meter["bb1"])
                data_meter["Criteria"] = data_meter["score"] * data_meter["Area"]
                idy = data_meter.groupby(['image'])['Criteria'].transform(
                    max) == data_meter['Criteria']
                meter_data_subset = data_meter[idy]
                meter_data_subset2 = meter_data_subset[meter_data_subset['class'] == "Car"]
                l_m_subset = len(np.unique(meter_data_subset2['image']))
                if l_m_subset == 0:
                    Output = Output.reset_index(drop=True)
                    Output.loc[0] = ['Car', 'Not Found']
                else:
                    meter_data_subset2['id'] = meter_data_subset2["image"].str.split(
                        claim_no1, n=0, expand=True)[1]
                    car_meter_data = pd.merge(car_data_subset2, meter_data_subset2[[
                                              'id', 'class', 'score', 'bb0', 'bb1', 'bb2', 'bb3']], how='inner', on='id')
                    l_car_meter = len(np.unique(car_meter_data['image']))
                    if l_car_meter == 0:
                        Output = Output.reset_index(drop=True)
                        Output.loc[0] = ['Car', 'Not Found']
                    else:
                        car_meter_data2 = car_meter_data[[
                            'id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x']]
                        car_final = car_meter_data2[[
                            'id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x']]
                        car_final["Area"] = (
                            car_final["bb2_x"] - car_final["bb0_x"]) * (car_final["bb3_x"] - car_final["bb1_x"])
                        #l_car_final = len(np.unique(car_final['image']))
    else:
        print("inside the else of merge_car_meter--------------------------")
        car_final = pd.DataFrame(columns= ['id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x', 'Area'], dtype=object)
        Output = Output.reset_index(drop=True)
        Output.loc[0] = ['Car', 'Not Found']
        
    if car_final is None:  
        print("car_final is None-------------------------")
        car_final = pd.DataFrame(columns= ['id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x', 'Area'], dtype=object)

    #print("#####################################car_final length##############", l_car_final)
    #print("Output from car*********", Output)
    print("Car_final--------------------------------------",car_final)
    #car_final.to_csv('car_final.csv', index = False)
    return(car_final, Output)


def merge_car_dent(car_final, data_dent, Output, data_part):
    #print("*********Output_car_for_dent******", Output)
    #print("***********In merge car dent", data_dent)
    dent_list = None
    dent_part = None
    l_d = len(np.unique(data_dent['image']))
    if l_d == 0:
        Output = Output.reset_index(drop=True)
        Output.loc[len(Output.index) + 1] = ['Dent', 'Not Found']
    else:
        data_dent1 = Damage_Intersection(data_dent)
        data_dent1['id'] = data_dent1["image"].str.split(
            claim_no1, n=0, expand=True)[1]
        car_dent = pd.merge(data_dent1, car_final[[
                            'id', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x']], on='id', how='inner')
        #print("**********car dent image", car_dent)
        if len(np.unique(car_dent['id'])) != 0:
            car_dent = car_dent.assign(Flag2=car_dent.apply(Flag_creation100, axis=1))
            car_dent2 = car_dent[car_dent["Flag2"] == 1]
            if len(np.unique(car_dent2['id'])) != 0:
                dent_list = pd.DataFrame((np.unique(car_dent2['id'])))
                dent_list["Damage_Type"] = "Dent"
                print("&&&& Dent List", dent_list)
                dent_list = dent_list.rename(columns={dent_list.columns[0]: "id"})
                Output = Output.append(dent_list, ignore_index=True)
                car_dent3 = car_dent2[['id', 'class', 'score', 'bb0', 'bb1', 'bb2', 'bb3']]
                dent_part = damage_part(data_part, car_dent3)
            else:
                Output = Output.reset_index(drop=True)
                Output.loc[len(Output.index) + 1] = ['Dent', 'Not Found']
        else:
            Output = Output.reset_index(drop=True)
            Output.loc[len(Output.index) + 1] = ['Dent', 'Not Found']
    #print("$$$$$$$$$$$$$$$$$ Output from dentlist", Output)
    #dent_part.to_csv("dent_part.csv", index = False)
    return(dent_list, Output, dent_part)


def merge_car_scratch(car_final, data_scratch, Output, data_part):
    scratch_list = None
    scratch_part = None 
    #data_scratch.to_csv("data_scratch.csv", index = False)
    l_s = len(np.unique(data_scratch['image']))
    if l_s == 0:
        Output = Output.reset_index(drop=True)
        Output.loc[len(Output.index) + 1] = ['Scratch', 'Not Found']
    else:
        data_scratch1 = Damage_Intersection(data_scratch)
        #data_scratch1.to_csv("data_scratch1.csv", index = False)
        data_scratch1['id'] = data_scratch1["image"].str.split(
            claim_no1, n=0, expand=True)[1]
        car_scratch = pd.merge(data_scratch1, car_final[[
                               'id', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x']], on='id', how='inner')
        if len(np.unique(car_scratch['id'])) > 0 :
            car_scratch = car_scratch.assign(Flag2=car_scratch.apply(Flag_creation100, axis=1))
            #car_scratch.to_csv("car_scratch.csv", index = False)
            car_scratch2 = car_scratch[car_scratch["Flag2"] == 1]
            if len(np.unique(car_scratch2['id'])) != 0:
                scratch_list = pd.DataFrame((np.unique(car_scratch2['id'])))
                scratch_list["Damage_Type"] = "Scratch"
                scratch_list = scratch_list.rename(columns={scratch_list.columns[0]: "id"})
                print("&&&& Scratch List", scratch_list)
                Output = Output.append(scratch_list, ignore_index=True)
                car_scratch3 = car_scratch2[['id', 'class', 'score', 'bb0', 'bb1', 'bb2', 'bb3']]
                #car_scratch3.to_csv("car_scratch3.csv", index=False)
                scratch_part = damage_part(data_part, car_scratch3)
            else:
                Output = Output.reset_index(drop=True)
                Output.loc[len(Output.index) + 1] = ['Scratch', 'Not Found']
        else:
            Output = Output.reset_index(drop=True)
            Output.loc[len(Output.index) + 1] = ['Scratch', 'Not Found']
    #print("$$$$$$$$$$$$$$$$$ Output from scratchlist", Output)
    #scratch_part.to_csv("scrach_part.csv", index=False)
    return(scratch_list, Output, scratch_part)

def merge_car_crack(car_final, data_crack, Output):
    #print("##########Output", Output)
    data_crack = data_crack[data_crack['class'].isin(["Bro_WS", "Bro_HL", "Bro_TL"])]
    crack_list = None
    car_crack3 = None
    l_crack = len(np.unique(data_crack['image']))
    if l_crack == 0:
        Output = Output.reset_index(drop=True)
        Output.loc[len(Output.index) + 1] = ['Crack', 'Not Found']
    else:
        data_crack1 = Damage_Intersection(data_crack)
        data_crack1['id'] = data_crack1["image"].str.split(claim_no1, n=0, expand=True)[1]
        #print('data_crack:', data_crack)
        crack_subset = data_crack1[data_crack1['score'] > 0]
        #print('crack_subset', crack_subset)
        crack_subset2 = crack_subset[crack_subset['class'].isin(["Bro_WS", "Bro_HL", "Bro_TL"])]
        #print('crack_subset2', crack_subset2)
        car_crack = pd.merge(crack_subset2, car_final[['id', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x']], on='id', how='inner')
        #print('car_crack', car_crack)
        if len(np.unique(car_crack['id'])) != 0:
            car_crack = car_crack.assign(Flag2=car_crack.apply(Flag_creation2, axis=1))
            car_crack2 = car_crack[car_crack["Flag2"] == 1]
            #print('car_crack2', car_crack2)
            if len(np.unique(car_crack2['id'])) != 0:
                crack_list = pd.DataFrame((np.unique(car_crack2['id'])))
                crack_list["Damage_Type"] = "Crack"
                crack_list = crack_list.rename(columns={crack_list.columns[0]: "id"})
                #print("&&&& Crack List", crack_list)
                Output = Output.append(crack_list, ignore_index=True)
                car_crack3 = car_crack2[['id', 'class', 'score', 'bb0', 'bb1', 'bb2', 'bb3']]
            else:
                Output = Output.reset_index(drop=True)
                #print("OUTPUT_CRACK**********", Output)
                Output.loc[len(Output.index) + 1] = ['Crack', 'Not Found']
        else:
            Output = Output.reset_index(drop=True)
            Output.loc[len(Output.index) + 1] = ['Crack', 'Not Found']
    #print("$$$$$$$$$$$$$$$$$ Output from cracklist", Output)
    return(crack_list, Output, car_crack3)



def merge_car_angle(car_final, data_angle):
    l_angle_final = 0
    car_angle = 0
    l_angle = None
    l_angle = len(np.unique(data_angle['image']))
    if l_angle != 0:
        
        data_angle['id'] = data_angle["image"].str.split(
            claim_no1, n=0, expand=True)[1]
        #print('data_angle:', data_angle)
        ida = data_angle.groupby(['image'])['score'].transform(
                max) == data_angle['score']
        angle_data_subset = data_angle[ida]
        #print('data_angle_subset##########:', angle_data_subset[['id' , 'score']])
        #print('----car_final---')
        car_angle = pd.merge(angle_data_subset, car_final[['id']], how = 'inner', on = 'id' )
        car_angle1 = car_angle[['class']]
        l_angle_final = len(np.unique(car_angle['class'])) 
        #car_angle.to_csv("car_angle.csv", index=False)
        #print('car_angle', car_angle1)
    return(l_angle_final, car_angle)

#---------------------------------Part_Side_Logic-----------------------------------#

def check180(car_angle,Breakin_id):
    avl_angle=list(car_angle['class'].value_counts().index)
    a = set(['FrontAngle_L','BackAngle_R']).issubset(avl_angle)
    b = set(['FrontAngle_R','BackAngle_L']).issubset(avl_angle)
    check180_data = []
    if a == True and b == False:        
        available = 1
        #c ='10'
        check180_data.append([a,b])
    elif b == True and a == False:
        available = 1
        #c = '01'
        check180_data.append([a,b])
    elif (a == True and b == True):
        available = 1
        #c = '11'
        check180_data.append([a,b])
    else:
        available = 0
        #c = '00'
        check180_data.append([a,b])
        
    check_180_df = pd.DataFrame(check180_data, columns = ['a','b'], dtype=object) #,index = False)
    check_180_df['Breakin_id'] = Breakin_id
    
    check_180_df.to_csv(str(Breakin_id) + '_check_180_df_' + str(present_time1) + '.csv')
    print("check_180_df=============== :", check_180_df)
    copy_to_blob_custom(str(Breakin_id) + '_check_180_df_' + str(present_time1),'check180')     
    os.remove(str(Breakin_id) + '_check_180_df_' + str(present_time1) + '.csv')
    
    # check_180_df.to_csv(str(Breakin_id) + '_check_180' +'.csv')
    
    # copy_to_blob_custom(str(Breakin_id) + '_check_180','check180')
        
    # os.remove(str(Breakin_id) + '_check_180' +'.csv')
    print("check_180_df===", check_180_df)
    print("available==", available)
    return available


def parts_angle_check_1(car_angle,data_part,Part_Angle_Master,sidecover_1, Breakin_id):    
    Part_Angle_Master["unique_Id"] = Part_Angle_Master["Angle"] +"_" + Part_Angle_Master["Parts"]
    sidecover_1["unique_Id"] = sidecover_1["Angle"] +"_" + sidecover_1["Parts"]
    part_angle_merge=pd.merge(data_part,car_angle, on="image", how="left")
    
    part_angle_merge.to_csv(str(Breakin_id) + '_part_angle_merge_' + str(present_time1) + '.csv')
    print("part_angle_merge=============== :", part_angle_merge)
    copy_to_blob_custom(str(Breakin_id) + '_part_angle_merge_' + str(present_time1),'partanglemerge')     
    os.remove(str(Breakin_id) + '_part_angle_merge_' + str(present_time1) + '.csv')
    
    part_angle_merge["unique_Id"] = part_angle_merge["class_y"] +"_" + part_angle_merge["class_x"]
    part_angle_merge=part_angle_merge.dropna(how='any', subset=['unique_Id'])
    print("This is part_angle_merge: ", part_angle_merge)
    avl_angle=list(car_angle['class'].value_counts().index)
    print("avl_angle=============== :", avl_angle)
    sides = ["Front","Back","SideL","SideR"] 
    avl_sides = common(avl_angle,sides)
    print("avl_sides=============== :", avl_sides)
    if len(avl_sides)==4:
        Expected_No_of_Parts=[]
        Detected_No_of_Parts=[]
        for i in avl_sides:
            start_letter = i + "_"   
            m = []
            m = [x for x in sidecover_1["unique_Id"] if x.startswith(start_letter)]        
            Expected_No_of_Parts.append(len(m))
            d = []
            d = [y for y in part_angle_merge["unique_Id"] if y.startswith(start_letter)]         
            d1 = np.unique(d)        
            d2 = common(d1, m)
            Detected_No_of_Parts.append(len(d2))
            print("In side "+str(i)+ ","+"out of total " +str(len(m)) + " parts, "+str(len(d2))+ " parts are Captured") 
        avl_sides_1 = pd.DataFrame({'Angle': avl_sides,'Expected_No_of_Parts': Expected_No_of_Parts,'Detected_No_of_Parts': Detected_No_of_Parts}, dtype=object)            
            
        avl_sides_1['Breakin_id'] = Breakin_id
        avl_sides_1['check'] = np.where((avl_sides_1['Expected_No_of_Parts'] == avl_sides_1['Detected_No_of_Parts']), 1, 0)
        
        avl_sides_1.to_csv(str(Breakin_id) + '_avl_sides_1_' + str(present_time1) + '.csv')
        print("avl_sides_1=============== :", avl_sides_1)
        copy_to_blob_custom(str(Breakin_id) + '_avl_sides_1_' + str(present_time1),'avlsides')     
        os.remove(str(Breakin_id) + '_avl_sides_1_' + str(present_time1) + '.csv')
        
        # avl_sides_1.to_csv(str(Breakin_id) + '_avl_sides_50'+  + '.csv')
        # print("avl_sides_1=============== :", avl_sides_1)
        # copy_to_blob_custom(str(Breakin_id) + '_avl_sides_50','avlsides')     
        # os.remove(str(Breakin_id) + '_avl_sides_50'+  + '.csv')
        
        check_1 = np.unique(avl_sides_1['check']) 
        print("....check count :",len(check_1))        
        
        print("....check value ======= :",check_1[0])
        if ( len(check_1)==1 and check_1[0]==1):
            proper_angle_image = 1
            print("11111111111111111111111")
        else:
            proper_angle_image = 0
            print("222222222222222222")
    else:
        proper_angle_image=0
        
    return proper_angle_image

def merge_car_angle_1(car_final, data_angle,data_part,master,sidecover_1,Breakin_id):
    #car_final.to_csv("car_final_15Dec.csv",index=False)
    l_angle_final = 0
    car_angle = 0
    l_angle = None
    l_angle = len(np.unique(data_angle['image']))
    if l_angle != 0:
        
        data_angle['id'] = data_angle["image"].str.split(
            claim_no1, n=0, expand=True)[1]
        print('data_angle:', data_angle)
        ida = data_angle.groupby(['image'])['score'].transform(
                max) == data_angle['score']
        angle_data_subset = data_angle[ida]
        print('data_angle_subset##########:', angle_data_subset[['id' , 'score']])
        #angle_data_subset.to_csv("angle_data_subset_15Dec.csv", index = False)
        car_angle = pd.merge(angle_data_subset, car_final[['id']], how = 'inner', on = 'id' )
        
        car_angle.to_csv(str(Breakin_id) + '_car_angle_' + str(present_time1) + '.csv')
        print("car_angle=============== :", car_angle)
        copy_to_blob_custom(str(Breakin_id) + '_car_angle_' + str(present_time1),'mergecarangle')     
        os.remove(str(Breakin_id) + '_car_angle_' + str(present_time1)  + '.csv')
        
        status_180 = check180(car_angle,Breakin_id)
        car_angle1 = car_angle[['class']]
        l_angle_final = len(np.unique(car_angle['class']))
        #car_angle.to_csv("car_angle_15Dec.csv", index=False)
        #car_angle.to_csv("car_angle.csv", index=False)
        
        print('car_angle', car_angle1)
        proper_angle_image = parts_angle_check_1(car_angle,data_part,master,sidecover_1, Breakin_id)
    return(l_angle_final, car_angle,status_180,proper_angle_image)



#---------------------------------Part_Side_Logic-----------------------------------#





def dent_scratch_final(part_damage_side, data_FP):
    part_damage_side7_4 = pd.DataFrame(columns=['id', 'Damage_Type'], dtype=object)
    part_damage_side7_3 = None
    #print("part_damage_side:", part_damage_side)
    List_Damage_ID = None
    data_FP = data_FP.add_suffix('_FP')
    #part_side_logic = pd.read_csv("Angle_Part_Tagging3.csv")

    part_damage_side1 =  pd.merge(part_damage_side, part_side_logic, left_on = 'class_x', right_on = 'Parts' , how = 'left')
    #parts = ['foglamp',	'frontwindshieldglass',	'headlamp','mirrorrearview', 'numberplate','rearwindshieldglass',	'rimwheel',	'taillamp',	'tyre',	'stickerblack	', 'stephanie', 'emblem',	'doorhandle',	'sensor',	'hook']
    part_damage_side1['Damage_ID'] = part_damage_side1['id'].astype(str) + part_damage_side1['class_y'].astype(str) + (part_damage_side1['score_y']).astype(str) + (part_damage_side1['bb0_y']).astype(str) + (part_damage_side1['bb1_y']).astype(str) + (part_damage_side1['bb2_y']).astype(str) + (part_damage_side1['bb3_y']).astype(str)
    #part_damage_side1.to_csv("check2.csv", index=False)
    #part_damage_side_subset =  part_damage_side1[part_damage_side1.class_x.isin(parts)]
    part_damage_side_subset = part_damage_side1[part_damage_side1['class_x'].isin(['EMBLEM','FOGLAMP','FRONTWINDSHIELDGLASS','HEADLAMP','MIRRORREARVIEW','NUMBERPLATE','REARWINDSHIELDGLASS',	'RIMWHEEL','TAILLAMP','TYRE','DOORHANDLE','STICKERBLACK','SENSOR','HOOK','STEPHANIE'])]
    part_damage_side_subset = part_damage_side_subset[part_damage_side_subset["Flag"] == 1]
    #part_damage_side_subset.to_csv("check3.csv", index=False)
    if len( part_damage_side_subset) > 0:
        part_damage_side_subset = part_damage_side_subset.assign(Flag_Internal = part_damage_side_subset.apply(Flag_creation4, axis=1))
        #part_damage_side_subset.to_csv("check1.csv", index=False)
        part_damage_side_subset1 = part_damage_side_subset[(part_damage_side_subset['Flag_Internal'] == 1)]
        part_damage_side_subset11 = part_damage_side_subset[(part_damage_side_subset['part_damage_dist'] >= 80)]
        #part_damage_side_subset11.to_csv("check4.csv", index=False)
        part_damage_side_subset1 =part_damage_side_subset1.append(part_damage_side_subset11, ignore_index=True)
        #part_damage_side_subset1.to_csv("check5.csv", index=False)
        List_Damage_ID = np.unique(part_damage_side_subset1[['Damage_ID']])
        part_damage_side2 =  part_damage_side1[~part_damage_side1.Damage_ID.isin(List_Damage_ID)]
        part_damage_side2 = part_damage_side2[~part_damage_side2.class_x.isin(['EMBLEM','FOGLAMP','FRONTWINDSHIELDGLASS','HEADLAMP','MIRRORREARVIEW','NUMBERPLATE','REARWINDSHIELDGLASS',	'RIMWHEEL','TAILLAMP','TYRE','DOORHANDLE','STICKERBLACK','SENSOR','HOOK','STEPHANIE'])]
        #part_damage_side2.to_csv("check6.csv", index=False)
    else :
        part_damage_side2 = part_damage_side1
        #part_damage_side2.to_csv("check7.csv", index=False)
        part_damage_side2 = part_damage_side2[~part_damage_side2.class_x.isin(['EMBLEM','FOGLAMP','FRONTWINDSHIELDGLASS','HEADLAMP','MIRRORREARVIEW','NUMBERPLATE','REARWINDSHIELDGLASS',	'RIMWHEEL','TAILLAMP','TYRE','DOORHANDLE','STICKERBLACK','SENSOR','HOOK','STEPHANIE'])]

    part_damage_side4 =  pd.merge(part_damage_side2, data_FP, left_on = 'image_x', right_on = 'image_FP' , how = 'left')
    part_damage_side4['Damage_ID'] = part_damage_side4['id'].astype(str) + part_damage_side4['class_y'].astype(str) + (part_damage_side4['score_y']).astype(str) + (part_damage_side4['bb0_y']).astype(str) + (part_damage_side4['bb1_y']).astype(str) + (part_damage_side4['bb2_y']).astype(str) + (part_damage_side4['bb3_y']).astype(str) 
    #art_damage_side4.to_csv('part_damage_side4.csv')    
    
    if len(part_damage_side4) > 0:
        part_damage_side4 = part_damage_side4.assign(Flag_FP = part_damage_side4.apply(Flag_creation5, axis=1))    
        part_damage_side4 = part_damage_side4.assign(Interaction_FP_Damage = part_damage_side4.apply(Flag_creation8, axis=1))
    
        part_damage_side4["bb0_z_FP"] = part_damage_side4[["bb0_FP", "bb0_y"]].max(axis=1)
        part_damage_side4["bb1_z_FP"] = part_damage_side4[["bb1_FP", "bb1_y"]].max(axis=1)
        part_damage_side4["bb2_z_FP"] = part_damage_side4[["bb2_FP", "bb2_y"]].min(axis=1)
        part_damage_side4["bb3_z_FP"] = part_damage_side4[["bb3_FP", "bb3_y"]].min(axis=1)
        part_damage_side4["inter_area_FP"] = (part_damage_side4["bb2_z_FP"] - part_damage_side4["bb0_z_FP"]) * (part_damage_side4["bb3_z_FP"] - part_damage_side4["bb1_z_FP"])
        part_damage_side4["damage_dist_FP"] = 100*part_damage_side4["inter_area_FP"]/part_damage_side4["damage_area"]    
        part_damage_side4 = part_damage_side4.assign(damage_fp_flag = part_damage_side4.apply(Flag_creation6, axis=1))    
        part_damage_side4["Area_FP"] = part_damage_side4["Interaction_FP_Damage"] + part_damage_side4["damage_fp_flag"]    
        part_damage_side4 = part_damage_side4.assign(Area_FP1 = part_damage_side4.apply(Flag_creation12, axis=1))     
        part_damage_side4["FINAL_FLAG_FP"] = part_damage_side4["Area_FP1"] + part_damage_side4['Flag_FP']    
        part_damage_side4_subset = part_damage_side4[(part_damage_side4['FINAL_FLAG_FP'] > 0)]
        Damage_id = np.unique(part_damage_side4_subset[['Damage_ID']])
        part_damage_side5 = part_damage_side4[~part_damage_side4.Damage_ID.isin(Damage_id)]
        #print('part_damage_side5====', part_damage_side5)
        
        if len(part_damage_side5) > 0:
            part_damage_side5 = part_damage_side5.assign(Flag_Part_Damage_Internal = part_damage_side5.apply(Flag_creation4, axis=1)) 
            part_damage_side6 = part_damage_side5[['id', 'class_y', 'score_y', 'class_x','score_x', 'class', 'score', 'part_damage_dist', 'Flag', 'Flag_Part_Damage_Internal']]
            part_damage_side6 = part_damage_side6[part_damage_side6['class_y'].notnull()]
        
            part_damage_side7 = part_damage_side6[(part_damage_side6['Flag'] > 0)]
            part_damage_side7_2  = part_damage_side7.drop_duplicates(subset=['id', 'class_y', 'score_y', 'class_x','score_x', 'class', 'score', 'part_damage_dist'], inplace=False)        

            part_damage_side7_3 = pd.merge(part_damage_side7_2, part_side_logic, left_on = 'class_x', right_on = 'Parts' , how = 'left')
        
        if part_damage_side7_3 is None:
            part_damage_side7_3 = pd.DataFrame(columns =['id','class_y','score_y','class_x','score_x','class','score','part_damage_dist','Flag','Flag_Part_Damage_Internal','Parts','Side','Priority'], dtype=object)
                
        if len(part_damage_side7_3) > 0:
            part_damage_side7_3 = part_damage_side7_3.assign(Flag_Part=part_damage_side7_3.apply(Flag_creation3, axis=1))
            #part_damage_side7_3.to_csv("check8.csv", index=False)
            part_damage_side7_3_1 = part_damage_side7_3[(part_damage_side7_3.Flag_Part == 1)]
            part_damage_side7_3_2 = part_damage_side7_3[part_damage_side7_3['Side'].isnull()]
            part_damage_side7_3 = part_damage_side7_3_1.append(part_damage_side7_3_2)
            #part_damage_side7_3.to_csv("check8_2.csv", index=False)
            part_damage_side7_3['Parts_Side'] = part_damage_side7_3['class_x'].astype(str) + part_damage_side7_3['Side']
            part_damage_side7_4 = part_damage_side7_3[['id', 'class_y', 'score_y', 'class_x','score_x', 'class', 'score', 'part_damage_dist']]
            part_damage_side7_4 = part_damage_side7_4.rename(index=str, columns={part_damage_side7_4.columns[5]: "Side"})
            #print("list of variables", list(part_damage_side_subset))
        else :
            part_damage_side7_4 = pd.DataFrame(columns=['id', 'Damage_Type'], dtype=object)
    else :
        part_damage_side7_4 = pd.DataFrame(columns=['id', 'Damage_Type'], dtype=object)
    
    #print('part damage side ======================')
    #print(part_damage_side7_4)
    #part_damage_side7_4.to_csv("part_damage_side7_4.csv")

    return(part_damage_side7_4)




def Flag_creation20(row):
    if row["Damage_Type"] == "Dent" and row["Type"] == "Metal" and row["part_damage_dist"] > 15 :
        return "REPLACE"
    elif  row["Damage_Type"] == "Dent" and row["Type"] == "Metal" and row["part_damage_dist"] <= 15:
        return "REPAIR"
    elif  row["Damage_Type"] == "Scratch" and row["Type"] == "Metal":
        return "REPAIR"
    elif  row["Type"] == "Glass" :
        return "REPLACE"
    elif  row["Damage_Type"] == "Scratch" and row["Type"] == "Plastic":
        return "REPLACE"
    elif  row["Damage_Type"] == "Dent" and row["Type"] == "Plastic":
        return "REPLACE"
    else:        
        return "Check"


def Flag_creation_Last(row):
    if row["Image_Name"] == 'Dent' and row["Damage_Type"]=='Not Found':
        return 1
    elif row["Image_Name"] == 'Scratch' and row["Damage_Type"]=='Not Found':
        return 1
    elif row["Image_Name"] == 'Crack' and row["Damage_Type"]=='Not Found':
        return 1
    
    elif row["Final_Flag_1"] == 1 and row["Damage_Type"]=='Car not found':
        return 1
    
    elif row["Final_Flag_1"] == 1 and row["Damage_Type"]=='Image is blur':
        return 1
    
    
    #-------------------------    
    # elif row["Image_Name"].contain('CHASSIS_NO') and row["Damage_Type"].contains('Car not found|Image is blur'):
    #     return 1
    
    # elif row["Image_Name"].contain('ODOMETER_READING') and row["Damage_Type"].contains('Car not found|Image is blur'):
    #     return 1
    
    # elif row["Image_Name"].contain('WINDSHIELD') and row["Damage_Type"].contains('Car not found|Image is blur'):
    #     return 1
    
    
    # elif row["Image_Name"].contain('RPM_READING') and row["Damage_Type"].contains('Car not found|Image is blur'):
    #     return 1
    
    # elif row["Image_Name"].contain('VEHICAL_REGISTRATION_NO') and row["Damage_Type"].contains('Car not found|Image is blur'):
    #     return 1
    
    
    # df.loc[df['Name'].str.contains('Andy|Andrew'),'Andy'] = 1
    # elif row["Image_Name"].isna() and row["Damage_Type"] =='Not Found':
    #     print("Image is null")
    #     return 1
    
    else:
        return 0


def final_count(final_output):
    
    #final_output_n = final_output
    #final_output.to_csv("final_output_17Feb.csv", index=False)
    #Damage_Type,Final_Flag,Side,Type,class_x,id,part_damage_dist,score,score_x,score_y,tmp
    
    #to remove negative profile
    #print('final_output_n before=====', final_output_n)
    #final_output_n = final_output_n[(final_output_n.Image_Name != 'np') | (final_output_n.Damage_Type != 'Found')]
    #print('final_output_n after=====', final_output_n)
    if len(final_output) == 0: 
        print("final output is empty---------------------------------------")
        final_output_n = pd.DataFrame(columns = ['id', 'Damage_Type', 'score_y', 'class_x', 'score_x', 'Side', 'score', 'part_damage_dist'], dtype=object)
        #final_output_n = final_output_n.loc[len(final_output_n)+1] = ['Summary', "No Damage Found", '', '', '', '', '', '', '']
    else:
        final_output['part'] = final_output['class_x'] + final_output['Side']
        final_output1 = final_output.assign(Flag_filter=final_output.apply(Flag_creation6_2, axis=1))
        #final_output1.to_csv("final_output1.csv", index=False)
        final_output1 = final_output1[final_output1['Flag_filter'] == 0]
        final_output1 = final_output1.drop(['Flag_filter'], axis=1)
        #final_output1.to_csv("RESULT1.csv", index=False)
    
    
    
        #final_output_check = final_output[final_output['score_y'] > 0] 
        #length3 = len(np.unique(final_output_n.id.str.slice(0, 7)))
        length3 = len(final_output1)
        #print("************length3*********** :" , length3)
    
        if length3 > 0 :
            f1 =  final_output1[final_output1['Damage_Type'].isin(["Dent", "Scratch"])]
            f2 =  final_output1[final_output1['Damage_Type'].isin(["Broken", "Bro_WS"])]
            f2_2 =  final_output1[final_output1['Damage_Type'].isin(["np"])]
            f2_3 =  final_output1[final_output1['Damage_Type'].isin(["Not Found"])]
            f2_4 =  final_output1[final_output1['Damage_Type'].isin(["Found"])]
            #f3 = len(final_output1) - len(f1) - len(f2) -len(f2_2) - len(f2_3)
            f3 = len(final_output1) - len(f1) - len(f2) -len(f2_2) - len(f2_3) -len(f2_4)
            f5 = final_output1[final_output1['Damage_Type'].isin(["No Car Image Found"])]
            
            length1 = 0
            length2 = 0
            length4 = 0
            length5 = 0
            length4 = f3
            length5 = len(f5)
            length6 = len(f2_3)
            
            if len(f1) > 0 :
                length1 = len(np.unique(f1["part"]))
            if len(f2) > 0 :
                length2 = len(np.unique(f2["part"]))
            
            final_output1 = final_output1.drop(['part'], axis=1)
            #print("length1 : ", length1)
            #print("length2 : ", length2)
            #print("length4 : ", length4)
            final_output1 = final_output1.reset_index(drop=True)
            
            print('----final_output1----', final_output1)
            #final_output1.to_csv('final_output1_17feb.csv', index = False)
            print('====list of final_output1---', list(final_output1))

            final_output1 = final_output1[['id', 'Damage_Type', 'Final_Flag', 'Side', 'Type', 'class_x', 'part_damage_dist', 'score', 'score_x', 'score_y', 'tmp']]
            
            if length1 == 0 and length2 ==0 and length6 == 0 and length5 == 1:
                final_output1 = final_output1 
                
            elif length4 > 0:
                final_output1.loc[len(final_output1)+1] = ['Summary', "Less than 8 images have car/few images are blur", '', '', '', '', '', '', '' ,'' ,''] 
            elif length2 > 0 and length1 > 3:
                 final_output1.loc[len(final_output1)+1] = ['Summary', "Crack found/two or more parts are damaged",'','', '', '', '', '', '', '','']
            elif length1 > 3 and length2 == 0:
                final_output1.loc[len(final_output1)+1] = ['Summary', "Two or more parts are damaged", '', '', '', '', '', '', '', '','']
            elif length2 > 0 and length1 <= 3:
                final_output1.loc[len(final_output1)+1] = ['Summary', "Crack found", '', '', '', '',  '', '', '', '','']
            elif length1 <= 3 and length2 == 0 and length5 != 1 and len(f2_4) ==0:
                final_output1.loc[len(final_output1)+1] = ["Summary", 'Less than two parts are damaged', '', '', '', '', '', '', '', '', '']
            elif len(f2_4)>0:
                final_output1.loc[len(final_output1)+1] = ['Summary', "Vehicle is not as per u/w guidelines", '', '', '', '', '', '', '', '', '']
            else:
                final_output1.loc[len(final_output1)+1] = ['Summary', "Check", '', '', '', '', '', '', '', '', '']
                
            final_output1 = final_output1.reset_index(drop=True)
            #final_output1.loc[len(final_output1)+1] = ['Summary', Summary, '', '', '', '', '', '']
            #final_output1.to_csv("RESULT2.csv", index=False)
    
            final_output_n = final_output1
        #print("Summary")
        
        #final_output_m = final_output_n[(final_output_n['id'] == 'np') & (final_output_n['Damage_Type'] == 'Found')]
        #final_output_m = np.logical_and(final_output_n[final_output_n['id'] == "np"], final_output_n[final_output_n['Damage_Type'] == 'Found'])
    
        #if len(final_output_m) > 0:
            #final_output1 = final_output1.reset_index(drop=True)
            #final_output1.loc[len(final_output1)+1] = ['Summary', "Negative Profile", '', '', '', '', '', '', '', '', '']
        #final_output_n = final_output1
        
        
    return(final_output_n)


def Repair_Replace(final_output, car_angle):
    #final_output.to_csv("final_output_temp.csv")
    if len(final_output) != 0:
        #final_output = pd.DataFrame(columns = ['id', 'Damage_Type', 'score_y', 'class_x', 'score_x', 'Side', 'score', 'part_damage_dist'])
        final_output_n = final_output
        #final_output.to_csv("final_output.csv", index=False)
        final_output1 = final_output[~final_output['class_x'].isin([ 'F_hood_L',	 'F_hood_R',	 'F_Bmper_R',	 'F_Bmper_L',	 'F_Lower',	 'FA_hood',	 'FA_Bumper_Lower',	 'FA_Bumper_Upper',	 'Side_upper',	 'Side_lower',	 'Side_tyer', 'BA_Backdoor_upper', 'BA_Bumper_upper',	 'BA_Bumper_lower',	'Back_Backdoor', 'Back_Numberplate_lower', 'Back_Bumper_upper', 'Back_Bumper_lower'])]
        final_output1 = final_output1.assign(Flag_filter=final_output1.apply(Flag_creation6_2, axis=1))
        #final_output1.to_csv("final_output1.csv", index=False)
        final_output1 = final_output1[final_output1['Flag_filter'] == 0]
        final_output1 = final_output1.drop(['Flag_filter'], axis=1)
        #final_output1.to_csv("RESULT1.csv", index=False)
        Data1 = final_output1[final_output1['Damage_Type'].isin(["Scratch", "Dent", "Bro_WS", "Bro_HL", "Bro_TL"])]
        Data2 = final_output1[final_output1['Damage_Type'].isin(["Found", 'Not Found',"Car not found", "Image is blur", "No Car Image Found"])]
        #Data2.to_csv('Data2.csv',index =False)
        Angle1 = car_angle
        Data1_Angle = None
        if len(Data1) > 0:
            Data1_Angle =  pd.merge(Data1 , Angle1[['id', 'class']], left_on = 'id', right_on = 'id' , how = 'inner')
            Data1_Angle = Data1_Angle.drop(['Side'],axis = 1)
            Data1_Angle = Data1_Angle.rename(columns={"class": "Side"})
            Data1_Angle['class_x'] = Data1_Angle['class_x'].fillna("0")
            temp = Data1_Angle.class_x.fillna(value= "0")
            print("temp***********", temp)
            Metal = ['FENDERFRONT','FRONTDOOR','REARDOOR','HOODASSY','EMBLEM','SIDEBODYPANEL','BACKDOOR','RIMWHEEL']
            Plastic = ['BUMPERFRONT','BUMPERREAR',	'GRILLBUMPERLOWER',	'GRILLFRONT',	'FOGLAMP',	'MIRRORREARVIEW',	'STICKERBLACK',	'DOORHANDLE',	'SENSOR',	'TYRE']
            Data1_Angle['Type'] = pd.np.where(temp.str.contains("0"),"Glass",
                       pd.np.where(temp.isin(Metal), "Metal",
                                   pd.np.where(temp.isin(Plastic), "Plastic","Glass")))
        Data1_Angle = pd.DataFrame(Data1_Angle)
        if len(Data1_Angle) == 0:
            Data1_Angle = pd.DataFrame(columns= ["id", 'Damage_Type', 'score_y', 'class_x', 'score_x',
                                                 'score', 'part_damage_dist', 'Side', 'Type', 'Final_Flag'], dtype=object)
        else:
            Data1_Angle = Data1_Angle.assign(Final_Flag=Data1_Angle.apply(Flag_creation20, axis=1))
            #Data1_Angle = Data1_Angle.assign(Final_Flag=Data1_Angle.apply(Flag_creation20, axis=1))
    
    else:
        Data1_Angle = pd.DataFrame(columns= ["id", 'Damage_Type', 'score_y', 'class_x', 'score_x', 'score', 'part_damage_dist', 'Side', 'Type', 'Final_Flag'], dtype=object)
        Data1_Angle.loc[0] = ['Summary', 'No Car Image Found', '', '', '', '', '', '', '', '']
        Data2 = pd.DataFrame(columns= ["id", 'CLAIM_REF_NO', 'Damage_Type', 'score_y', 'class_x', 'score_x', 'Side', 'score', 'part_damage_dist', 'tmp'], dtype=object)

    #print("Dta1_Angle*******", Data1_Angle)

    #Data1_Angle.to_csv("Data1_Angle.csv")     
    return(Data1_Angle, Data2)





def Model(Dataframe):

    DZIRE = Dataframe['VEH_MODEL'].str.contains('IRE') | Dataframe['VEH_MODEL'].str.contains('ire')
    BALENO = Dataframe['VEH_MODEL'].str.contains('LENO') | Dataframe['VEH_MODEL'].str.contains('leno')
    ALTO = Dataframe['VEH_MODEL'].str.contains('ALTO') | Dataframe['VEH_MODEL'].str.contains('lto')
    WAGONR = Dataframe['VEH_MODEL'].str.contains('WAGON') | Dataframe['VEH_MODEL'].str.contains('agon') | Dataframe['VEH_MODEL'].str.contains('agonR')

    #creating a new column named MODEL with all entries as swift
    Dataframe['MODEL'] = "SWIFT"

    #replacing the entries of MODEL column accordingly
    Dataframe['MODEL'][BALENO]='BALENO'
    Dataframe['MODEL'][ALTO]='ALTO'
    Dataframe['MODEL'][WAGONR]='WAGON R'
    Dataframe['MODEL'][DZIRE]='SWIFT DZIRE'
    
    return(Dataframe)



#----to remove duplicate entries of single possible parts like windshield glass and etc.---------
def remove_duplicate(result):
    
    result['Count'] = result.groupby('PART').cumcount()+1
    df_A1 = result.loc[(result['PART'] == 'HOODASSY') & (result['Count'] >= 2)]
    df_B1 = result.loc[(result['PART'] == 'FRONTWINDSHIELDGLASS') & (result['Count'] >= 2)]
    df_C1 = result.loc[(result['PART'] == 'BUMPERFRONT') & (result['Count'] >= 2)]
    df_D1 = result.loc[(result['PART'] == 'BUMPERREAR') & (result['Count'] >= 2)]
    df_E1 = result.loc[(result['PART'] == 'GRILLBUMPERLOWER') & (result['Count'] >= 2)]
    df_F1 = result.loc[(result['PART'] == 'FRONTWINDSHIELDGLASS') & (result['Count'] >= 2)]
    df_G1 = result.loc[(result['PART'] == 'GRILLFRONT') & (result['Count'] >= 2)]
    df_H1 = result.loc[(result['PART'] == 'REARWINDSHIELDGLASS') & (result['Count'] >= 2)]


    df_comb = [df_A1, df_B1, df_C1, df_D1, df_E1, df_F1, df_G1, df_H1]
    df_remove = pd.concat(df_comb, ignore_index=True)

    result = pd.merge(result, df_remove, how='outer', indicator=True).query('_merge=="left_only"').drop('_merge',1)
    result = result.drop(['Count'], axis =1)
    return(result)


def Diff(li1, li2): 
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2] 
    return li_dif



#-----------------------------------------------------------------------------------------


def common(a,b): 
    c = [value for value in a if value in b] 
    return c


def final_blur_output(df_name):
    #df_name.to_csv('output_blur_to_analyse.csv', index = False)
    df_name[['Image_Name', 'Extension']] = df_name.id.str.rsplit('.', 1, expand = True) 
    original_name = list(df_name['id'].str.contains('rt'))
    
    if False in original_name:
            df_name['id'] = df_name['Image_Name'] + '.' + df_name['Extension']  
            
    if True in original_name:
        df_name[['image1', 'angle']] = df_name.Image_Name.str.split("_rt", expand=True) 
        df_name['id'] = df_name['image1'] + '.' +  df_name['Extension'] 
        
    #df_name[['no_of_image', 'extension']] = df_name.second.str.split(".", expand=True)
    
    #df_name['id'] = df_name['First'] + "_" + df_name['no_of_image'] + '.png'
    
    
    df_name = df_name[['id', 'Damage_Type', 'score_y', 'class_x', 'score_x', 'Side', 'score','part_damage_dist']]
    df_name.drop_duplicates(subset ="id", inplace = True)
    
    return(df_name)

def Final_Car(car_final):
    
    #car_final.to_csv('car1.csv', index =False)
    #print('-----Inside Final car function.....car final is', car_final)
    
    if len(car_final) == 0:
        car_final = pd.DataFrame(columns = ['id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x', 'Area', 'id_1'], dtype=object)
    
    else:
        
        #original_name = list(car_final['id'].str.contains('rt'))
        
        car_final[['Image_Name', 'Extension']] = car_final.id.str.rsplit('.', 1, expand = True) 
        original_name = list(car_final['id'].str.contains('rt'))
        
        #car_final.to_csv('car_final_to_analyse.csv', index =False)
        if False in original_name:
            car_final['id_1'] = car_final['Image_Name'] + '.' + car_final['Extension']  
            
        if True in original_name:
            car_final[['image1', 'angle']] = car_final.Image_Name.str.split("_rt", expand=True) 
            car_final['id_1'] = car_final['image1'] + '.' +  car_final['Extension'] 
    
    #car_final.to_csv('car2.csv', index =False)
    car_final = car_final.sort_values('score_x', ascending=False).drop_duplicates(['id_1']) 
    car_final = car_final[['id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x', 'Area', 'id_1']]
    #car_final.to_csv('car3.csv', index =False)
    car_final.reset_index(inplace = True, drop = True)
    return(car_final)

def Final_Result(dffff):
    print('dffff', dffff)
    
    if len(dffff) == 0:
        dffff = pd.DataFrame(columns = ['id', 'Damage_Type', 'score_y' ,'class_x', 'score_x', 'Side', 'score', 'part_damage_dist', 'Image_Name', 'Extension'], dtype=object)
        
    else:
        Not_found = dffff[dffff.Damage_Type == 'Not Found']
        df1 = dffff[~dffff.index.isin(Not_found.index)]
        
        if len(df1) != 0:
            df1[['Image_Name', 'Extension']] = df1.id.str.rsplit('.', 1, expand = True) 
            original_name = list(df1['id'].str.contains('rt'))
            
            #car_final.to_csv('car_final_to_analyse.csv', index =False)
            if False in original_name:
                df1['id'] = df1['Image_Name'] + '.' + df1['Extension']
                
            if True in original_name:
                df1[['image1', 'angle']] = df1.id.str.split("_rt", expand=True) 
                df1['id'] = df1['image1'] + '.' +  df1['Extension'] 
                print('dffff 2', df1)
                
            df2 = Not_found
            temp =[df1, df2]
            out = pd.concat(temp, ignore_index=True)
        else:          
            out = Not_found
            out['Image_Name'] = ''
            out['Extension'] = ''

    return(out)


def Final_Result1(dffff):
    print('dffff', dffff)
    if len(dffff) == 0:
        dffff = pd.DataFrame(columns = ['id', 'Damage_Type', 'score_y' ,'class_x', 'score_x', 'Side', 'score', 'part_damage_dist'], dtype=object)
        
    else:
        
        dama_list = list(np.unique(dffff['Damage_Type']))
        
        if len(dama_list) == 1 and dama_list[0] == 'Not Found':
            dffff = dffff
        else:
            dffff[['Image_Name', 'Extension']] = dffff.id.str.rsplit('.', 1, expand = True) 
            original_name = list(dffff['id'].str.contains('rt'))
            
            #car_final.to_csv('car_final_to_analyse.csv', index =False)
            if False in original_name:
                dffff['id'] = dffff['Image_Name'] + '.' + dffff['Extension']  
                
            if True in original_name:
                dffff[['image1', 'angle']] = dffff.id.str.split("_rt", expand=True) 
                dffff['id'] = dffff['image1'] + '.' +  dffff['Extension'] 
            print('dffff 2', dffff)
            
            dffff = dffff.drop(['Image_Name', 'Extension'], axis =1)
    
    return(dffff)

def remove_rotation_flag_from_file_list(modified_images):
    print("inside the rotation flag remove function---------------")
    sub_list = ["_rt_90", "_rt_270"] 
    print("sub_list", sub_list)
    for i in sub_list:
        print(i)
        res1 = [x.replace(str(i), "") for x in modified_images]
        path = res1
        print(res1)
    return res1

def images_to_process_2(test_image_path, Breakin_id):
    print('-----in images to process---------------')
    global car_final_1
    global car_final_2
    global car_final_3
    global Final_NBlur1_1
    global Final_NBlur1_2
    global Final_NBlur1_3
    global car_not_found_1
    global car_not_found_2
    global car_not_found_3
    global Output_1
    global Output_2
    global Output_3
    global odo_check
    global RPM_check
    global Engine_On_and_Off_check
    global Chassis_check
    print('here 1111')
    odo_check = 0
    # RPM_check = 0
    RPM_check = 1
    Engine_On_and_Off_check = 0
    Chassis_check = 0
    print('here 2222')
    
    
    Final_NBlur1_1 = []
    Final_NBlur1_2 = []
    Final_NBlur1_3 = []
    
    car_not_found_1 = {}
    car_not_found_2 = {}
    car_not_found_3 = {}
    print('here 3333')
    
    column_names_out = ["id", "Damage_Type"] #, "c"]
    Output_1 = pd.DataFrame(columns = column_names_out, dtype=object)
    Output_2 = pd.DataFrame(columns = column_names_out, dtype=object)
    Output_3 = pd.DataFrame(columns = column_names_out, dtype=object)
    #Output_1 = pd.DataFrame(columns=['id', 'Damage_Type'])
    # Output_2 = pd.DataFrame(columns=['id', 'Damage_Type'])
    # Output_3 = pd.DataFrame(columns=['id', 'Damage_Type'])

    received_images_list = os.listdir(test_image_path)
    print("received images========================",len(received_images_list))
    print("Received images list name ==============",received_images_list)
    
    test_imgs_subset, test_imgs, test_imgs_not_car = filter_images(test_image_path)
    test_imgs_rm = [x.replace((test_image_path + '/'), '') for x in test_imgs]
    print("test_imgs =========================",len(test_imgs))
    print("test_imgs==========================",test_imgs)
    print("test_imgs_rm =========================",len(test_imgs_rm))
    print("test_imgs_rm =========================",test_imgs_rm)
    print("after the filter images ====================================")
    
    # NBlur1, Output_Blur1, Blur1 = check_blur(test_imgs)
    NBlur1, Output_Blur1, Blur1 = check_blur(test_imgs_subset)
    print("NBlur1=========================",len(NBlur1) )
    print("NBlur1 ==================", NBlur1)
    print("Blur1================================",len(Blur1))
    
    car_final_1 = pd.DataFrame(columns= ['id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x', 'Area', 'id_1'], dtype=object)
    car_final_2 = pd.DataFrame(columns= ['id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x', 'Area', 'id_1'], dtype=object)
    car_final_3 = pd.DataFrame(columns= ['id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x', 'Area', 'id_1'], dtype=object)
    
    
    #extracting desired image lists
    NBlur1_car = [x for x in NBlur1 if x not in test_imgs_not_car]
    #NBlur1_RPM = common(test_imgs_not_car, NBlur1)
    NBlur1_RPM = test_imgs_not_car 
    print("NBlur1_car::", len(NBlur1_car))
    print("NBlur1_car:", NBlur1_car)
    print("NBlur1_RPM_length::", len(NBlur1_RPM))
    print("NBlur1_RPM:", NBlur1_RPM)
    if len(test_imgs_subset)>0:
        for item in test_imgs_subset:
            #path = os.getcwd()
            im = Image.open(item).convert('RGB')
            im.save(str(item))
        
        test_images = NBlur1_car
        print("test_images 1=========================",test_images )
        
        if len(NBlur1) != 0:
            print("inside the NBlur1 != 0 !!!!!!!!!!!!!!!!!!!!!")
            
            data_odo = tf_od_pred("odo", NBlur1_RPM)
            #data_odo_1 = tf_od_pred("odo_1", NBlur1_RPM)
            #data_odo.to_csv('temp/3Mar_data_odo_'+ str(Breakin_id) + '.csv', index = False)
        
            #data_temp = data_odo[data_odo['class'] == 'RpmMeter']
            #data_temp.to_csv('temp/3Mar_data_temp_'+ str(Breakin_id) + '.csv', index = False)
            
            #list_of_RPM = data_temp['image']
            #print('list_of_RPM:',list_of_RPM)
            
            
            
            if ('RpmMeter' in list(data_odo['class'])):
                RPM_check = 1
                print("RpmMeter found=================")
                
            if ('OdoMeter' in list(data_odo['class'])):
                odo_check = 1
                print("OdoMeter found====================")
                    
            if ('EngineChassis' in list(data_odo['class'])):
                Chassis_check = 1
                #data_RPM = tf_od_pred("RPM", list_of_RPM)
                #data_RPM.to_csv('temp/3Mar_data_RPM_' + str(Breakin_id) + '.csv', index = False)
                print("EngineChassis found================")
                
                        # if ('Engine_on' in list(data_RPM['class'])):
                        #     Engine_On_and_Off_check = 1
                        #     print("Engine_on found=================")  
            #if Chassis_check == 1 and RPM_check ==1 and odo_check ==1:
            if Chassis_check == 1 and odo_check ==1:
                data_car = tf_od_pred("car", NBlur1_car)

                #data_car.to_csv('temp/3Mar_data_car_'+ str(Breakin_id) + '.csv', index = False)
    
                data_odo = tf_od_pred("odo", NBlur1_car)
                #data_odo.to_csv('3Mar_data_odo_car_'+ str(Breakin_id) + '.csv', index = False)
    
                data_meter = data_odo[data_odo['class'] == 'Car']
                # data_car = tf_od_pred("car", NBlur1_car)
                # data_car.to_csv('17thFeb_data_car_1.csv', index = False)
                # data_meter = tf_od_pred("meter", NBlur1)
                # data_meter.to_csv('17thFeb_data_meter_1.csv', index = False)
                #------------------------------------
                Output = pd.DataFrame(columns=['id', 'Damage_Type'], dtype=object)
                
                car_final_1, Output_1 = merge_car_meter(data_car, data_meter, Output)
                # car_final_1.to_csv("temp/car_final_before_1_" + str(Breakin_id) + ".csv",index = False)
                car_final_1= Final_Car(car_final_1)
                #car_found_1 = np.unique(car_final_1["id"])
                
                #Final_NBlur1_1 = list(car_final_1['image'])
                
                #car_not_found_1 = set(received_images_list) - set(car_found_1)
                #car_final_1.to_csv("temp/car_final_1_" + str(Breakin_id) + ".csv", index = False)
                #Output_Blur1_1 = check_blur_output(test_imgs,Blur1,flag_2333)
                
                # car_final = car_final
                # Final_NBlur1 = list(car_final['image'])
                # Output =  Output
                # Blur1 = Blur1
                # car_not_found = car_not_found
                # NBlur1 = NBlur1
                # Output_Blur1 = Output_Blur1
                
                if len(car_final_1.index) == len(NBlur1_car):
                    print("all images processed successfully ==================")
                    
                else:
                    print("inside the else after car final 1 ==============")
                    car_for_rt_90 = [test_image_path + '/' + i for i in car_final_1['id_1']]
                    print("car_for_rt_90=====================",car_for_rt_90)
                    #test_images = set(test_images) - set(car_final_1['image'])
                    test_images = set(test_images) - set(car_for_rt_90)
                    print("test_images 2=========================",test_images )
                    test_images_rm = [x.replace((test_image_path + "/"), '') for x in test_images]
                    print("test_images2_1 ==============", test_images_rm)
                    for q in test_images_rm:
                        #img_name = q.split('.')[0]
                        #ext = q.split('.')[1]
                        print("Current Working Directory=============", os.getcwd())
                        img_name, ext = os.path.splitext(q)
                        print( "img_name =======================", img_name)
                        print("ext =============================", ext)
                        img_rt_90 = rotate_img(test_image_path+'/'+q, 90)
                        print("current path ==================",os.getcwd())
                        img_rt_90.save(str(test_image_path) + '/' + str(img_name)+'_rt_90' + str(ext))
                        
                        # img_rt_270 = rotate_img(test_image_path+'/'+q, 270)
                        # img_rt_270.save(str(test_image_path) + '/' + str(img_name)+'_rt_270'+ str(ext))
                    
                    modified_images = os.listdir(test_image_path)
                                    
                    rotated_90_images = [s for s in modified_images if "_rt_90" in s]
                    rotated_90_images = [test_image_path + '/' + i for i in rotated_90_images]
                    
                    data_car = tf_od_pred("car", rotated_90_images)
                    #data_car.to_csv("temp/3Mar_data_car_2_" + str(Breakin_id) + ".csv", index = False)
                    data_odo = tf_od_pred("odo", rotated_90_images)
                    data_meter = data_odo[data_odo['class'] == 'Car']
                    #data_meter.to_csv('temp/3Mar_data_meter_2' +str(Breakin_id) + '.csv', index = False)
                    #------------------------------------
                    Output = pd.DataFrame(columns=['id', 'Damage_Type'], dtype=object)
                    car_final_2, Output_2 = merge_car_meter(data_car, data_meter, Output)
                    # car_final_2.to_csv("car_final_before_2.csv")
                    car_final_2 = Final_Car(car_final_2)
                    #car_final_2.to_csv("temp/car_final_2_" + str(Breakin_id) + ".csv", index = False)
                    #car_found_2 = np.unique(car_final_2["id"]) 
                    #Final_NBlur1_2 = list(car_final_2['image'])
                    #car_not_found_2 = set(test_images) - set(car_found_2['image'])
                    
                    #Output_Blur1_1 = check_blur_output(test_imgs,Blur1,flag_2333)
                    #NBlur1_2, Output_Blur1_2, Blur1_2 = check_blur(test_images)
                    #merge all car final and all output and add all car not found, final nblur1 
                    # car_final = car_final
                    # Final_NBlur1 = list(car_final['image']) #60
                    # Output =  Output
                    # Blur1 = Blur1
                    # car_not_found = car_not_found
                    # NBlur1 = NBlur1
                    # Output_Blur1 = Output_Blur1
                    
                    if len(car_final_2.index) == len(rotated_90_images):
                        print("all rt90 images processed successfully ==================") 
                    
                    else:
                        
                        print("inside the 270 rotation loop====================")
                        car_for_rt_270 = [test_image_path + '/' + i for i in car_final_2['id_1']]
                        print("car_for_rt_270=====================",car_for_rt_270)
                        #test_images = set(test_images) - set(car_final_2['image'])
                        test_images = set(test_images) - set(car_for_rt_270)
                        print("test_images 3=========================",test_images )
                        test_images_rm = [x.replace((test_image_path + "/"), '') for x in test_images]
                        print("test_images_rm 3_1=========================",test_images_rm )
                        modified_images = os.listdir(test_image_path)
                        #remove _rt_90
                        modified_images = remove_rotation_flag_from_file_list(modified_images)
                        for q in test_images_rm:
                            #img_name = q.split('.')[0]
                            #ext = q.split('.')[1]
                            
                            img_name, ext = os.path.splitext(q)
                            
                            # img_rt_90 = rotate_img(test_image_path+'/'+q, 90)
                            # img_rt_90.save(str(test_image_path) + '/' + str(img_name)+'_rt_90' + str(ext))
                            img_rt_270 = rotate_img(test_image_path+'/'+q, 270)
                            img_rt_270.save(str(test_image_path) + '/' + str(img_name)+'_rt_270'+ str(ext))
    
                        modified_images = os.listdir(test_image_path)
    
                        rotated_270_images = [s for s in modified_images if "_rt_270" in s]
                        rotated_270_images = [test_image_path + '/' + i for i in rotated_270_images]
                        data_car = tf_od_pred("car", rotated_270_images)
                        #data_car.to_csv('temp/3Mar_data_car_3' + str(Breakin_id) + '.csv', index = False)
                        data_odo = tf_od_pred("odo", rotated_270_images)
                        data_meter = data_odo[data_odo['class'] == 'Car']
                        #data_meter.to_csv('temp/3Mar_data_meter_3_' + str(Breakin_id) + '.csv', index = False)
                        #------------------------------------
                        Output = pd.DataFrame(columns=['id', 'Damage_Type'], dtype=object)
                        car_final_3, Output_3 = merge_car_meter(data_car, data_meter, Output)
                        #car_final_3.to_csv("car_final_before_3.csv")
                        car_final_3 = Final_Car(car_final_3)
                        #car_final_3.to_csv("car_final_3_" + str(Breakin_id) +".csv", index = False)
                        #car_found_3 = np.unique(car_final_3["id"]) 
                        #Final_NBlur1_3 = list(car_final_3['image'])
                        #car_not_found_3 = set(test_images) - set(car_found_3['image'])
                        #NBlur1_new, Output_Blur1, Blur1_new = check_blur(test_imgs)
                        
                        # car_final = car_final
                        # Final_NBlur1 = list(car_final['image']) #60
                        # Output =  Output
                        # Blur1 = Blur1
                        # car_not_found = car_not_found
                        # NBlur1 = NBlur1
                        # Output_Blur1 = Output_Blur1
                        
                        if len(car_final_3.index) == len(rotated_270_images):
                            print("all rt270 images processed successfully ==================") 
                    
                        else:
                            car_final = pd.DataFrame(columns = ['id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x', 'Area', 'id_1'], dtype=object)
                            Final_NBlur1 = []
                            Output = pd.DataFrame(columns=['id', 'Damage_Type'], dtype=object)
                            
                            Blur1 = Blur1
                            car_not_found = {}
                            
                            car_angle = 0
                            #print("preparing final output with blur only")
                            final_output = (Output_Blur1)
                            #print("final_output66666", final_output)
                #merge here
                #car_final_1.to_csv("car_final_1.csv", index = False)
                #car_final_2.to_csv("car_final_2.csv", index = False)
                #car_final_3.to_csv("car_final_3.csv", index = False)    
                if car_final_1 is None:
                    print("carfinal1  is none =====================")
                    car_final_1 = pd.DataFrame(columns= ['id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x', 'Area', 'id_1'], dtype=object)
                
                if car_final_2 is None:
                    print("carfinal2  is none =====================")
                    car_final_2 = pd.DataFrame(columns= ['id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x', 'Area', 'id_1'], dtype=object)
                
                if car_final_3 is None:
                    print("carfinal3  is none =====================")
                    car_final_3 = pd.DataFrame(columns= ['id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x', 'Area', 'id_1'], dtype=object)
    
    
                car_final = pd.concat([car_final_1, car_final_2,car_final_3], ignore_index=True)
                car_final = Final_Car(car_final)
                #car_final.to_csv("temp/car_final_concated_" + str(Breakin_id) + ".csv", index =False)
                Final_NBlur1 = list(car_final['image'])
                Output = pd.concat([Output_1, Output_2,Output_3], ignore_index=True)
                car_found = np.unique(car_final["id_1"]) 
                car_not_found = set(received_images_list) - set(car_found)
            
            
            else:
                print("Engine Chassis/Odometer is not found")
                car_final = pd.DataFrame(columns= ['id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x', 'Area', 'id_1'], dtype=object)
                #car_final.loc[0] = ['Car', 'Not Found','','','','','','','','']
            
                Final_NBlur1 = []
                Output = pd.DataFrame(columns=['id', 'Damage_Type'], dtype=object)
                Output.loc[0] = ['Car', 'Not Found']
                Blur1 = []
                car_not_found = {}
                NBlur1 = []
                
                Output_Blur1 = pd.DataFrame(columns=['id', 'Damage_Type', 'score_y', 'class_x', 'score_x', 'Side', 'score', 'part_damage_dist'], dtype=object)
                Output_Blur1.loc[0] = ['Image', 'Does not exist', '', '', '', '', '', '' ]
                
                    
                    # else:
                    #     print("Engine Chassis is not found")
                    #     car_final = pd.DataFrame(columns= ['id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x', 'Area', 'id_1'])
                    #     #car_final.loc[0] = ['Car', 'Not Found','','','','','','','','']
                    
                    #     Final_NBlur1 = []
                    #     Output = pd.DataFrame(columns=['id', 'Damage_Type'])
                    #     Output.loc[0] = ['Car', 'Not Found']
                    #     Blur1 = []
                    #     car_not_found = {}
                    #     NBlur1 = []
                        
                    #     Output_Blur1 = pd.DataFrame(columns=['id', 'Damage_Type', 'score_y', 'class_x', 'score_x', 'Side', 'score', 'part_damage_dist'])
                    #     Output_Blur1.loc[0] = ['Image', 'Does not exist', '', '', '', '', '', '' ]
                
                # else:
                #     print("ODO Meter is not found")
                #     car_final = pd.DataFrame(columns= ['id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x', 'Area', 'id_1'])
                #     #car_final.loc[0] = ['Car', 'Not Found','','','','','','','','']
                
                #     Final_NBlur1 = []
                #     Output = pd.DataFrame(columns=['id', 'Damage_Type'])
                #     Output.loc[0] = ['Car', 'Not Found']
                #     Blur1 = []
                #     car_not_found = {}
                #     NBlur1 = []
                    
                #     Output_Blur1 = pd.DataFrame(columns=['id', 'Damage_Type', 'score_y', 'class_x', 'score_x', 'Side', 'score', 'part_damage_dist'])
                #     Output_Blur1.loc[0] = ['Image', 'Does not exist', '', '', '', '', '', '' ]
                    
            # else:
            #     print("RPM Meter is not found")
            #     car_final = pd.DataFrame(columns= ['id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x', 'Area', 'id_1'])
            #     #car_final.loc[0] = ['Car', 'Not Found','','','','','','','','']
            
            #     Final_NBlur1 = []
            #     Output = pd.DataFrame(columns=['id', 'Damage_Type'])
            #     Output.loc[0] = ['Car', 'Not Found']
            #     Blur1 = []
            #     car_not_found = {}
            #     NBlur1 = []
                
            #     Output_Blur1 = pd.DataFrame(columns=['id', 'Damage_Type', 'score_y', 'class_x', 'score_x', 'Side', 'score', 'part_damage_dist'])
            #     Output_Blur1.loc[0] = ['Image', 'Does not exist', '', '', '', '', '', '' ]
                
        else:
            print("after filtering all images,rest images are checked blured ")
            #make dataframe of returns variable
            
            #make dataframe of returns variable
            car_final = pd.DataFrame(columns= ['id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x', 'Area'], dtype=object)
            #car_final.loc[0] = ['Car', 'Not Found']
            
            Final_NBlur1 = []
            Output = pd.DataFrame(columns=['id', 'Damage_Type'], dtype=object)
            Output.loc[0] = ['Car', 'Not Found']
            Blur1 = []
            car_not_found = {}
            NBlur1 = []
            
            Output_Blur1 = pd.DataFrame(columns=['id', 'Damage_Type', 'score_y', 'class_x', 'score_x', 'Side', 'score', 'part_damage_dist'], dtype=object)
            Output_Blur1.loc[0] = ['Image', 'Does not exist', '', '', '', '', '', '' ]
    
    
    
    else:
        print("Length of images after filtering is less than zero")
        
        #make dataframe of returns variable
        car_final = pd.DataFrame(columns= ['id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x', 'Area'], dtype=object)
        
        #car_final.loc[0] = ['Car', 'Not Found']
        
        Final_NBlur1 = []
        Output = pd.DataFrame(columns=['id', 'Damage_Type'], dtype=object)
        Output.loc[0] = ['Car', 'Not Found']
        Blur1 = []
        car_not_found = {}
        NBlur1 = []
        
        Output_Blur1 = pd.DataFrame(columns=['id', 'Damage_Type', 'score_y', 'class_x', 'score_x', 'Side', 'score', 'part_damage_dist'], dtype=object)
        Output_Blur1.loc[0] = ['Image', 'Does not exist', '', '', '', '', '', '' ]
        
    return(car_final, Final_NBlur1, Output, Blur1, car_not_found, NBlur1, Output_Blur1,odo_check,RPM_check,Engine_On_and_Off_check,Chassis_check)


# def images_to_process_1(test_image_path):
#     print('-----in images to process---------------')
#     global car_final_1
#     global car_final_2
#     global car_final_3
#     global Final_NBlur1_1
#     global Final_NBlur1_2
#     global Final_NBlur1_3
#     global car_not_found_1
#     global car_not_found_2
#     global car_not_found_3
#     global Output_1
#     global Output_2
#     global Output_3
#     car_final_1 = pd.DataFrame(columns= ['id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x', 'Area'], dtype=object)
#     car_final_2 = pd.DataFrame(columns= ['id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x', 'Area'], dtype=object)
#     car_final_3 = pd.DataFrame(columns= ['id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x', 'Area'], dtype=object)
#     Final_NBlur1_1 = []
#     Final_NBlur1_2 = []
#     Final_NBlur1_3 = []
#     car_not_found_1 = {}
#     car_not_found_2 = {}
#     car_not_found_3 = {}
#     Output_1 = pd.DataFrame(columns=['id', 'Damage_Type'])
#     Output_2 = pd.DataFrame(columns=['id', 'Damage_Type'])
#     Output_3 = pd.DataFrame(columns=['id', 'Damage_Type'])

#     received_images_list = os.listdir(test_image_path)
#     print("received images========================",len(received_images_list))
#     print("Received images list name ==============",received_images_list)
#     test_imgs = filter_images(test_image_path)
#     test_imgs_rm = [x.replace((test_image_path + '/'), '') for x in test_imgs]
#     print("test_imgs =========================",len(test_imgs))
#     print("test_imgs==========================",test_imgs)
#     print("test_imgs_rm =========================",len(test_imgs_rm))
#     print("test_imgs_rm =========================",test_imgs_rm)
#     print("after the filter images ====================================")
#     NBlur1, Output_Blur1, Blur1 = check_blur(test_imgs)
#     print("NBlur1=========================",len(NBlur1) )
#     print("NBlur1 ==================", NBlur1)
#     print("Blur1================================",len(Blur1))
#     if len(test_imgs)>0:
#         for item in test_imgs:
#             #path = os.getcwd()
#             im = Image.open(item).convert('RGB')
#             im.save(str(item))
        
#         test_images = NBlur1
#         print("test_images 1=========================",test_images )
#         if len(NBlur1) != 0:
#             print("inside the NBlur1 != 0 !!!!!!!!!!!!!!!!!!!!!")
#             data_car = tf_od_pred("car", NBlur1)
#             #data_car.to_csv('17thFeb_data_car_1.csv', index = False)
#             data_meter = tf_od_pred("meter", NBlur1)
#             #data_meter.to_csv('17thFeb_data_meter_1.csv', index = False)
#             #------------------------------------
#             Output = pd.DataFrame(columns=['id', 'Damage_Type'])
            
#             car_final_1, Output_1 = merge_car_meter(data_car, data_meter, Output)
#             #car_final_1.to_csv("car_final_before_1.csv",index = False)
#             car_final_1= Final_Car(car_final_1)
#             #car_found_1 = np.unique(car_final_1["id"])
            
#             #Final_NBlur1_1 = list(car_final_1['image'])
            
#             #car_not_found_1 = set(received_images_list) - set(car_found_1)
#             #car_final_1.to_csv("car_final_1.csv", index = False)
#             #Output_Blur1_1 = check_blur_output(test_imgs,Blur1,flag_2333)
            
#             # car_final = car_final
#             # Final_NBlur1 = list(car_final['image'])
#             # Output =  Output
#             # Blur1 = Blur1
#             # car_not_found = car_not_found
#             # NBlur1 = NBlur1
#             # Output_Blur1 = Output_Blur1
            
#             if len(car_final_1.index) == len(NBlur1):
#                 print("all images processed successfully ==================")
                
#             else:
#                 print("inside the else after car final 1 ==============")
#                 car_for_rt_90 = [test_image_path + '/' + i for i in car_final_1['id_1']]
#                 print("car_for_rt_90=====================",car_for_rt_90)
#                 #test_images = set(test_images) - set(car_final_1['image'])
#                 test_images = set(test_images) - set(car_for_rt_90)
#                 print("test_images 2=========================",test_images )
#                 test_images_rm = [x.replace((test_image_path + "/"), '') for x in test_images]
#                 print("test_images2_1 ==============", test_images_rm)
#                 for q in test_images_rm:
#                     #img_name = q.split('.')[0]
#                     #ext = q.split('.')[1]
#                     print("Current Working Directory=============", os.getcwd())
#                     img_name, ext = os.path.splitext(q)
#                     print( "img_name =======================", img_name)
#                     print("ext =============================", ext)
#                     img_rt_90 = rotate_img(test_image_path+'/'+q, 90)
#                     print("current path ==================",os.getcwd())
#                     img_rt_90.save(str(test_image_path) + '/' + str(img_name)+'_rt_90' + str(ext))
                    
#                     # img_rt_270 = rotate_img(test_image_path+'/'+q, 270)
#                     # img_rt_270.save(str(test_image_path) + '/' + str(img_name)+'_rt_270'+ str(ext))
                
#                 modified_images = os.listdir(test_image_path)
                                
#                 rotated_90_images = [s for s in modified_images if "_rt_90" in s]
#                 rotated_90_images = [test_image_path + '/' + i for i in rotated_90_images]
#                 data_car = tf_od_pred("car", rotated_90_images)
#                 #data_car.to_csv('17thFeb_data_car_2.csv', index = False)
#                 data_meter = tf_od_pred("meter", rotated_90_images)
#                 #data_meter.to_csv('17thFeb_data_meter_2.csv', index = False)
#                 #------------------------------------
#                 Output = pd.DataFrame(columns=['id', 'Damage_Type'])
#                 car_final_2, Output_2 = merge_car_meter(data_car, data_meter, Output)
#                 #car_final_2.to_csv("car_final_before_2.csv")
#                 car_final_2 = Final_Car(car_final_2)
#                 #car_final_2.to_csv("car_final_2.csv", index = False)
#                 #car_found_2 = np.unique(car_final_2["id"]) 
#                 #Final_NBlur1_2 = list(car_final_2['image'])
#                 #car_not_found_2 = set(test_images) - set(car_found_2['image'])
                
#                 #Output_Blur1_1 = check_blur_output(test_imgs,Blur1,flag_2333)
#                 #NBlur1_2, Output_Blur1_2, Blur1_2 = check_blur(test_images)
#                 #merge all car final and all output and add all car not found, final nblur1 
#                 # car_final = car_final
#                 # Final_NBlur1 = list(car_final['image']) #60
#                 # Output =  Output
#                 # Blur1 = Blur1
#                 # car_not_found = car_not_found
#                 # NBlur1 = NBlur1
#                 # Output_Blur1 = Output_Blur1
                
#                 if len(car_final_2.index) == len(rotated_90_images):
#                     print("all rt90 images processed successfully ==================") 
                
#                 else:
                    
#                     print("inside the 270 rotation loop====================")
#                     car_for_rt_270 = [test_image_path + '/' + i for i in car_final_2['id_1']]
#                     print("car_for_rt_270=====================",car_for_rt_270)
#                     #test_images = set(test_images) - set(car_final_2['image'])
#                     test_images = set(test_images) - set(car_for_rt_270)
#                     print("test_images 3=========================",test_images )
#                     test_images_rm = [x.replace((test_image_path + "/"), '') for x in test_images]
#                     print("test_images_rm 3_1=========================",test_images_rm )
#                     modified_images = os.listdir(test_image_path)
#                     #remove _rt_90
#                     modified_images = remove_rotation_flag_from_file_list(modified_images)
#                     for q in test_images_rm:
#                         #img_name = q.split('.')[0]
#                         #ext = q.split('.')[1]
                        
#                         img_name, ext = os.path.splitext(q)
                        
#                         # img_rt_90 = rotate_img(test_image_path+'/'+q, 90)
#                         # img_rt_90.save(str(test_image_path) + '/' + str(img_name)+'_rt_90' + str(ext))
#                         img_rt_270 = rotate_img(test_image_path+'/'+q, 270)
#                         img_rt_270.save(str(test_image_path) + '/' + str(img_name)+'_rt_270'+ str(ext))

#                     modified_images = os.listdir(test_image_path)

#                     rotated_270_images = [s for s in modified_images if "_rt_270" in s]
#                     rotated_270_images = [test_image_path + '/' + i for i in rotated_270_images]
#                     data_car = tf_od_pred("car", rotated_270_images)
#                     #data_car.to_csv('17thFeb_data_car_3.csv', index = False)
#                     data_meter = tf_od_pred("meter", rotated_270_images)
#                     #data_meter.to_csv('17thFeb_data_meter_3.csv', index = False)
#                     #------------------------------------
#                     Output = pd.DataFrame(columns=['id', 'Damage_Type'])
#                     car_final_3, Output_3 = merge_car_meter(data_car, data_meter, Output)
#                     #car_final_3.to_csv("car_final_before_3.csv")
#                     car_final_3 = Final_Car(car_final_3)
#                     #car_final_3.to_csv("car_final_3.csv", index = False)
#                     #car_found_3 = np.unique(car_final_3["id"]) 
#                     #Final_NBlur1_3 = list(car_final_3['image'])
#                     #car_not_found_3 = set(test_images) - set(car_found_3['image'])
#                     #NBlur1_new, Output_Blur1, Blur1_new = check_blur(test_imgs)
                    
#                     # car_final = car_final
#                     # Final_NBlur1 = list(car_final['image']) #60
#                     # Output =  Output
#                     # Blur1 = Blur1
#                     # car_not_found = car_not_found
#                     # NBlur1 = NBlur1
#                     # Output_Blur1 = Output_Blur1
                    
#                     if len(car_final_3.index) == len(rotated_270_images):
#                         print("all rt270 images processed successfully ==================") 
                
#                     else:
#                         car_final = pd.DataFrame(columns = ['id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x', 'Area'])
#                         Final_NBlur1 = []
#                         Output = pd.DataFrame(columns=['id', 'Damage_Type'])
                        
#                         Blur1 = Blur1
#                         car_not_found = {}
                        
#                         car_angle = 0
#                         #print("preparing final output with blur only")
#                         final_output = (Output_Blur1)
#                         #print("final_output66666", final_output)
#             #merge here
#             #car_final_1.to_csv("car_final_1.csv", index = False)
#             #car_final_2.to_csv("car_final_2.csv", index = False)
#             #car_final_3.to_csv("car_final_3.csv", index = False)    
#             if car_final_1 is None:
#                 print("carfinal1  is none =====================")
#                 car_final_1 = pd.DataFrame(columns= ['id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x', 'Area'])
            
#             if car_final_2 is None:
#                 print("carfinal2  is none =====================")
#                 car_final_2 = pd.DataFrame(columns= ['id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x', 'Area'])
            
#             if car_final_3 is None:
#                 print("carfinal3  is none =====================")
#                 car_final_3 = pd.DataFrame(columns= ['id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x', 'Area'])


#             car_final = pd.concat([car_final_1, car_final_2,car_final_3], ignore_index=True)
#             car_final = Final_Car(car_final)
#             #car_final.to_csv("car_final_concated_" + str(Breakin_id) + ".csv", index =False)
#             Final_NBlur1 = list(car_final['image'])
#             Output = pd.concat([Output_1, Output_2,Output_3], ignore_index=True)
#             car_found = np.unique(car_final["id_1"]) 
#             car_not_found = set(received_images_list) - set(car_found)
#         else:
#             print("after filtering all images,rest images are checked blured ")
#             #make dataframe of returns variable
            
#             #make dataframe of returns variable
#             car_final = pd.DataFrame(columns= ['id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x', 'Area'])
#             #car_final.loc[0] = ['Car', 'Not Found']
            
#             Final_NBlur1 = []
#             Output = pd.DataFrame(columns=['id', 'Damage_Type'])
#             Output.loc[0] = ['Car', 'Not Found']
#             Blur1 = []
#             car_not_found = {}
#             NBlur1 = []
            
#             Output_Blur1 = pd.DataFrame(columns=['id', 'Damage_Type', 'score_y', 'class_x', 'score_x', 'Side', 'score', 'part_damage_dist'])
#             Output_Blur1.loc[0] = ['Image', 'Does not exist', '', '', '', '', '', '' ]
#     else:
#         print("Length of images after filtering is less than zero")
        
#         #make dataframe of returns variable
#         car_final = pd.DataFrame(columns= ['id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x', 'Area'])
#         #car_final.loc[0] = ['Car', 'Not Found']
        
#         Final_NBlur1 = []
#         Output = pd.DataFrame(columns=['id', 'Damage_Type'])
#         Output.loc[0] = ['Car', 'Not Found']
#         Blur1 = []
#         car_not_found = {}
#         NBlur1 = []
        
#         Output_Blur1 = pd.DataFrame(columns=['id', 'Damage_Type', 'score_y', 'class_x', 'score_x', 'Side', 'score', 'part_damage_dist'])
#         Output_Blur1.loc[0] = ['Image', 'Does not exist', '', '', '', '', '', '' ]
      
        
#     return(car_final, Final_NBlur1, Output, Blur1, car_not_found, NBlur1, Output_Blur1)

# def images_to_process(test_image_path):
#     #print('-----in images to process---------------')
#     received_images_list = os.listdir(test_image_path)
#     #print('----list of received images with csv-----', received_images_list)
#     #print('====before policy info removal=====', len(received_images_list))
#     #received_images_list.remove('Policy_info.csv')

#     #only_images = getFiles(test_image_path)
#     #print('=====list of only images======', received_images_list)
#     #print('====after policy info removal=====', len(received_images_list))
#     for q in received_images_list:
        
#         #print('image apth is', str(test_image_path) + '/' + str(q))
#         #img_open = cv2.imread(str(test_image_path) + '/' + str(q))
#         #height, width, depth = img_open.shape
#         #print('======Rotating required images only=======')
#         #if width <= height:
#         if os.stat(test_image_path+'/'+q).st_size > 0:
#             img_name, ext = os.path.splitext(q)
            
#             img_rt_90 = rotate_img(test_image_path+'/'+q, 90)
#             img_rt_90.save(str(test_image_path) + '/' + str(img_name)+'_rt_90' + str(ext))
            
#             img_rt_270 = rotate_img(test_image_path+'/'+q, 270)
#             img_rt_270.save(str(test_image_path) + '/' + str(img_name)+'_rt_270'+ str(ext))
        
#     modified_images = os.listdir(test_image_path)
#     #print('----list of modified_images -----', modified_images)
#     #print('=======count of images after rotation=====', len(modified_images))
    
#     test_imgs = filter_images(test_image_path)
#     #print('----test_imgs-----------', test_imgs)
    
#     if len(test_imgs) > 0:
#         for item in test_imgs:
#             #path = os.getcwd()
#             im = Image.open(item).convert('RGB')
#             im.save(str(item))
#         NBlur1, Output_Blur1, Blur1 = check_blur(test_imgs)
        
#         Output_Blur1
        
#         print('Number of not blur = ',len(NBlur1), 'Number of blur', len(Blur1))
#     else:
#         NBlur1 = []
#         Output_Blur1 = pd.DataFrame(columns=['id', 'Damage_Type', 'score_y', 'class_x', 'score_x', 'Side', 'score', 'part_damage_dist'])
#         Output_Blur1.loc[0] = ['Image', 'Does not exist', '', '', '', '', '', '' ]
#         Blur1 = []
    
    
#     #print('------NNNBBBLllluurrrrr------', NBlur1)
    
#     if len(NBlur1) != 0:
#         #print("***********NBLUR NOT 0")
#         data_car = tf_od_pred("car", NBlur1)
#         #data_car.to_csv('17thFeb_data_car.csv', index = False)
#         data_meter = tf_od_pred("meter", NBlur1)
#         #data_meter.to_csv('17thFeb_data_meter.csv', index = False)
#         #------------------------------------
#         Output = pd.DataFrame(columns=['id', 'Damage_Type'])
#         car_final, Output = merge_car_meter(data_car, data_meter, Output)
#         #car_final.to_csv('17thFeb_car_final.csv', index =False)
#         #print('=======car_final_2=======', car_final)
#         #print('----$$$$---output----$$$----', Output)
#         #print('========NBlur1========', NBlur1)
        
#         #car_final_temp = car_final
        
#         #calling ''final_car'' function
#         car_final = Final_Car(car_final)
#         #car_final.to_csv('Final_Car.csv', index = False)
#         #print('==========Car_Final after function==============', car_final)
        
#         #print('=====car_final before car_found========', car_final)
#         car_found = np.unique(car_final["id_1"])
        
#         #print('=====list of only images======', received_images_list)
#         #print('=====list of images in car_final===', car_found)
        
#         car_not_found = set(received_images_list) - set(car_found)
        
#         #print('=========list of car_not_found==========', car_not_found)
        
#         #car_final.to_csv('car_final_2.csv', index = False)
        
#         l_car_final = len(np.unique(car_final['image']))
        
#         #print('======l_car_final=====', l_car_final)
#         #------------------------------------
#         Final_NBlur1 = list(car_final['image'])
#         #print('========Final_NBlur1========', Final_NBlur1)
        
#     else:
#         car_final = pd.DataFrame(columns = ['id', 'image', 'class_x', 'score_x', 'bb0_x', 'bb1_x', 'bb2_x', 'bb3_x', 'Area'])
#         Final_NBlur1 = []
#         Output = pd.DataFrame(columns=['id', 'Damage_Type'])
        
#         Blur1 = Blur1
#         car_not_found = []
        
#         car_angle = 0
#         #print("preparing final output with blur only")
#         final_output = (Output_Blur1)
#         #print("final_output66666", final_output)
        
#     return(car_final, Final_NBlur1, Output, Blur1, car_not_found, NBlur1, Output_Blur1)


def damage_update(df1):
    #df1.to_csv('damage_update.csv', index =False)
    #print('----inside damage update', df1)
    df1["tmp"] = df1['Damage_Type'] + '_' + df1['Side']
        
    #decoding of some code words of AI model 
    df1.loc[df1.Damage_Type == "Bro_HL", 'class_x'] = "HEADLAMP"
    df1.loc[df1.Damage_Type == "Bro_HL", 'Damage_Type'] = "Broken"
    df1.loc[df1.Damage_Type == "Bro_TL", 'class_x'] = "TAILLAMPREAR"
    df1.loc[df1.Damage_Type == "Bro_TL", 'Damage_Type'] = "Broken"
    df1.loc[df1.tmp == "Bro_WS_FrontAngle_R", 'class_x'] = "FRONTWINDSHIELDGLASS"
    df1.loc[df1.tmp == "Bro_WS_FrontAngle_L", 'class_x'] = "FRONTWINDSHIELDGLASS"
    df1.loc[df1.tmp == "Bro_WS_Front", 'class_x'] = "FRONTWINDSHIELDGLASS"
    df1.loc[df1.tmp == "Bro_WS_BackAngle_R", 'class_x'] = "REARWINDSHIELDGLASS"
    df1.loc[df1.tmp == "Bro_WS_BackAngle_L", 'class_x'] = "REARWINDSHIELDGLASS"
    df1.loc[df1.tmp == "Bro_WS_Back", 'class_x'] = "REARWINDSHIELDGLASS"
    
    df1.loc[df1.tmp == "Bro_WS_FrontAngle_R", 'Damage_Type'] = "Broken"
    df1.loc[df1.tmp == "Bro_WS_FrontAngle_L", 'Damage_Type'] = "Broken"
    df1.loc[df1.tmp == "Bro_WS_Front", 'Damage_Type'] = "Broken"
    df1.loc[df1.tmp == "Bro_WS_BackAngle_R", 'Damage_Type'] = "Broken"
    df1.loc[df1.tmp == "Bro_WS_BackAngle_L", 'Damage_Type'] = "Broken"
    df1.loc[df1.tmp == "Bro_WS_Back", 'Damage_Type'] = "Broken"
    
    return(df1)


def rotate_img(img_path, rt_degr):
    img = Image.open(img_path)
    return img.rotate(rt_degr, expand=1)


def init():                 
    #global rf_from_joblib
    global dentGraph
    global scratchGraph
    global crackGraph
    global carGraph
    global meterGraph
    global meterLRGraph
    global partGraph
    global npGraph
    global RPMGraph
    global odoGraph
    global odoGraph_1
    #global designLineGraph
    global fpdamagesGraph
    global model_folder
    global path_label_template
    global part_side_logic
    global Replace_final
    global metal_parts_with_D_ND
    global pdtextdent
    global pdtextcar
    global pdtextcrack
    #global pdtextdesignLine
    global pdtextfpdamages
    global pdtextmeter
    global pdtextmeterLR
    global pdtextnp
    global pdtextpart
    global pdtextscratch
    global pdtextRPM
    global pdtextodo
    #global pdtextodo_1
    global hitlim
    #global City_tag_file
    #global Repair_Data_all
    #global FINAL_TOP5_MODELS_REPLACE
    global master
    global sidecover_1

    
    #model_folder = "/mnt/batch/tasks/applications/breakin1.02020-11-10-08-23/models"
    
    #model_folder = "/mnt/batch/tasks/applications/breakin2.02021-03-08-14-45/models_breakin"
    
    #model_folder = "/mnt/batch/tasks/applications/breakin3.02021-03-09-03-10/models_breakin"
    
    #model_folder = "/mnt/batch/tasks/applications/breakin4.02021-04-08-00-00/models_breakin_7Apr"
    # model_folder = "/mnt/batch/tasks/applications/breakin5.02021-04-23-11-16/models_breakin_23Apr"
    #model_folder = "/mnt/batch/tasks/applications/breakin6.02021-06-07-12-07/models_breakin_7Jun"     
    #model_folder = 'models'
    # model_folder = 'Models_BreakIn'
    # model_folder = "/mnt/batch/tasks/applications/breakin6.02021-06-07-12-07/models_breakin_7Jun"
    model_folder = "/mnt/batch/tasks/applications/breakin8.02022-10-22-06-34/Models_Breakin_17Oct2022"
    model_folder = op.join(os.getcwd(), model_folder)

    part_side_logic = pd.read_csv(op.join(model_folder, 'Angle_Part_Tagging3.csv'))
    
    
    master = pd.read_csv(op.join(model_folder, 'Parts_vs_Angle.csv'))
    sidecover_1=pd.read_csv(op.join(model_folder, 'sidecover_1.csv'))

    #City_tag_file = pd.read_csv(op.join(model_folder, 'City_Tag.csv'))
    #Repair_Data_all = pd.read_csv(op.join(model_folder, "Top5_Repair.csv"))  
    #FINAL_TOP5_MODELS_REPLACE = pd.read_csv(op.join(model_folder, "FINAL_TOP5_MODELS_REPLACE.csv"))


    #metal_parts_with_D_ND = pd.read_csv(op.join(model_folder, 'metal_parts_with_D_ND.csv'))

    #Replace_final = pd.read_csv(op.join(model_folder, 'Replace_Final.csv'))

    #global claim_no1
    #load from some path .pbText files 
    pdtextdent = label_map_util.load_labelmap(op.join(model_folder, 'pascal_label_map_dent.pbtxt'))
    pdtextscratch = label_map_util.load_labelmap(op.join(model_folder,"pascal_label_map_scratch.pbtxt"))
    pdtextcrack = label_map_util.load_labelmap(op.join(model_folder,"pascal_label_map_crack.pbtxt"))
    pdtextcar = label_map_util.load_labelmap(op.join(model_folder,"pascal_label_map_car.pbtxt"))
    pdtextmeter = label_map_util.load_labelmap(op.join(model_folder,"pascal_label_map_meter.pbtxt"))
    pdtextmeterLR = label_map_util.load_labelmap(op.join(model_folder,"pascal_label_map_meterLR.pbtxt"))
    pdtextpart = label_map_util.load_labelmap(op.join(model_folder,"pascal_label_map_part.pbtxt"))
    pdtextnp = label_map_util.load_labelmap(op.join(model_folder,"pascal_label_map_np.pbtxt"))
    #pdtextdesignLine = label_map_util.load_labelmap(op.join(model_folder,"pascal_label_map_designLine.pbtxt"))
    pdtextfpdamages = label_map_util.load_labelmap(op.join(model_folder,"pascal_label_map_fpdamages.pbtxt"))
    pdtextRPM = label_map_util.load_labelmap(op.join(model_folder,"pascal_label_map_RPM.pbtxt"))
    pdtextodo = label_map_util.load_labelmap(op.join(model_folder,"pascal_label_map_odo.pbtxt"))
    #pdtextodo_1 = label_map_util.load_labelmap(op.join(model_folder,"pascal_label_map_odo_1.pbtxt"))
    #rf_from_joblib = joblib.load(op.join(model_folder,'Top5_model_pred.pkl'))  

    dentGraph, scratchGraph, crackGraph, carGraph, meterGraph, meterLRGraph, partGraph, fpdamagesGraph, npGraph, RPMGraph, odoGraph, odoGraph_1 = loadAllModels()


def copy_from_blob(file_name):
    try:
        #account_name = "devinstaclaimstorage"
        #account_key = "ibF92RsbmV1dGrilw+y/rmajWj86VJc2V+mD1T2iFQbqQsrZ4vnkxC4wWfeGF4fP5lKj4IfcH60H2oyFwh+WVw=="
        container_name = "breakininput"
        #block_blob_service = BlockBlobService(account_name='devinstaclaimstorage', account_key='ibF92RsbmV1dGrilw+y/rmajWj86VJc2V+mD1T2iFQbqQsrZ4vnkxC4wWfeGF4fP5lKj4IfcH60H2oyFwh+WVw==')
        connect_str = "DefaultEndpointsProtocol=https;AccountName=newbreakinstorage;AccountKey=kXz3auR20UJWRGmcnTE+KUeCRiFblk2dr0TcqMaBrehV8tQ7l3fpWIQtA4noLb0KEVRAeaOP/SKRcqU0g/JtbQ==;EndpointSuffix=core.windows.net"
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        local_path  = os.getcwd()
        file_name = file_name + ".zip"
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)
        full_path_to_file2 = os.path.join(local_path, file_name)
        with open(full_path_to_file2, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        #block_blob_service.get_blob_to_path( container_name, local_file_name, full_path_to_file2)
    except Exception as e:
        print(e)

def copy_to_blob(file_name):
    try:
        #account_name = "devinstaclaimstorage"
        #account_key = "ibF92RsbmV1dGrilw+y/rmajWj86VJc2V+mD1T2iFQbqQsrZ4vnkxC4wWfeGF4fP5lKj4IfcH60H2oyFwh+WVw=="
        print('inside_blob')
        container_name = "runtime"
        #block_blob_service = BlockBlobService(account_name='devinstaclaimstorage', account_key='ibF92RsbmV1dGrilw+y/rmajWj86VJc2V+mD1T2iFQbqQsrZ4vnkxC4wWfeGF4fP5lKj4IfcH60H2oyFwh+WVw==')
        connect_str = "DefaultEndpointsProtocol=https;AccountName=newbreakinstorage;AccountKey=kXz3auR20UJWRGmcnTE+KUeCRiFblk2dr0TcqMaBrehV8tQ7l3fpWIQtA4noLb0KEVRAeaOP/SKRcqU0g/JtbQ==;EndpointSuffix=core.windows.net"
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        local_path  = os.getcwd()
        file_name = str(file_name) + ".csv"
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)
        full_path_to_file2 = os.path.join(local_path, file_name)
        with open(full_path_to_file2, "rb") as data:
            print('uploading')
            blob_client.upload_blob(data)
        #block_blob_service.get_blob_to_path( container_name, local_file_name, full_path_to_file2)
    except Exception as e:
        print(e)
        
        
#---------------------Customize Function to Upload File in Azure Container ------------------------#

def copy_to_blob_custom(file_name,container_name):
    try:
        #account_name = "devinstaclaimstorage"
        #account_key = "ibF92RsbmV1dGrilw+y/rmajWj86VJc2V+mD1T2iFQbqQsrZ4vnkxC4wWfeGF4fP5lKj4IfcH60H2oyFwh+WVw=="
        print('inside_blob')
        container_name = container_name
        #block_blob_service = BlockBlobService(account_name='devinstaclaimstorage', account_key='ibF92RsbmV1dGrilw+y/rmajWj86VJc2V+mD1T2iFQbqQsrZ4vnkxC4wWfeGF4fP5lKj4IfcH60H2oyFwh+WVw==')
        connect_str = "DefaultEndpointsProtocol=https;AccountName=newbreakinstorage;AccountKey=kXz3auR20UJWRGmcnTE+KUeCRiFblk2dr0TcqMaBrehV8tQ7l3fpWIQtA4noLb0KEVRAeaOP/SKRcqU0g/JtbQ==;EndpointSuffix=core.windows.net"
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        local_path  = os.getcwd()
        file_name = str(file_name) + ".csv"
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)
        full_path_to_file2 = os.path.join(local_path, file_name)
        with open(full_path_to_file2, "rb") as data:
            print('uploading')
            blob_client.upload_blob(data)
        #block_blob_service.get_blob_to_path( container_name, local_file_name, full_path_to_file2)
    except Exception as e:
        print(e)



#---------------------Customize Function to Upload File in Azure Container ------------------------#


def insert_entity(status, result, breakinUniqueid):
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Inside update table")
    table_name = 'BreakinOutput'

    table_service = TableService(account_name = 'newbreakinstorage', account_key='kXz3auR20UJWRGmcnTE+KUeCRiFblk2dr0TcqMaBrehV8tQ7l3fpWIQtA4noLb0KEVRAeaOP/SKRcqU0g/JtbQ==')


    tableEntry = Entity()
    tableEntry.PartitionKey = 'Surveyor'
    global present_time
    present_time = str(datetime.datetime.now())
    
    replacements_dict = {'-': '_',
                     ' ': '_',
                     ':':'_',
                     '.':'_'}
    global present_time1
    present_time1 = present_time.translate(str.maketrans(replacements_dict))
    
    tableEntry.RowKey = breakinUniqueid + "_" + present_time
    tableEntry.BreakinID = breakinUniqueid
    print(status)

    #if status == "Completed":
     #   message1 = status #{"Status": status, "Result": json.loads(result)}
      #  message2 = json.loads(result)
    #else:
     #   print(result)
      #  message1 = status #{"Status": status, "Result": result}
       # message2 = result

    tableEntry.priority = 100
    tableEntry.status = 'Wait'
    #tableEntry.priority = 100
    tableEntry.result = 'Awaiting'

    table_service.insert_entity(table_name, tableEntry)


def update_table_status(status, result, breakinUniqueid):
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Inside update table")
    table_name = 'BreakinOutput'

    table_service = TableService(account_name = 'newbreakinstorage', account_key='kXz3auR20UJWRGmcnTE+KUeCRiFblk2dr0TcqMaBrehV8tQ7l3fpWIQtA4noLb0KEVRAeaOP/SKRcqU0g/JtbQ==')


    tableEntry = Entity()
    tableEntry.PartitionKey = 'Surveyor'
    tableEntry.RowKey = breakinUniqueid + "_" + present_time
    tableEntry.BreakinID = breakinUniqueid
    print(status)

    if status == "Completed":
        message1 = status #{"Status": status, "Result": json.loads(result)}
        message2 = json.loads(result)
    else:
        print(result)
        message1 = status #{"Status": status, "Result": result}
        message2 = result

    tableEntry.priority = 100
    tableEntry.status = json.dumps(message1)
    #tableEntry.priority = 100
    tableEntry.result = json.dumps(message2)

    table_service.update_entity(table_name, tableEntry)


#table_service = TableService(
  #  account_name=account_name, account_key=account_key)

def combine_flag(RPM_check,Chassis_check,odo_check):  
    RPM_check = 1
    final_recapture_flag = str(RPM_check) + "_"  + str(Chassis_check) + "_" + str(odo_check)
    final_recapture_flag = str(final_recapture_flag)
    print(final_recapture_flag)
    return final_recapture_flag

def check_existing(Breakin_id):
    blob_name = Breakin_id + '.csv'
    container_name = 'runtime'
    connect_str = "DefaultEndpointsProtocol=https;AccountName=newbreakinstorage;AccountKey=kXz3auR20UJWRGmcnTE+KUeCRiFblk2dr0TcqMaBrehV8tQ7l3fpWIQtA4noLb0KEVRAeaOP/SKRcqU0g/JtbQ==;EndpointSuffix=core.windows.net"
    blob = BlobClient.from_connection_string(conn_str=connect_str, container_name=container_name, blob_name=blob_name)
    exists = blob.exists()
    print(exists)
    
    if exists == True:
        print('Deleting the existing')
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        container_client = blob_service_client.get_container_client(container_name)
        container_client.delete_blobs(blob_name)
    else:
        print('Not existing')
        pass

def Check_existing_breakin(Breakin_id):
    
    table_service = TableService(connection_string='DefaultEndpointsProtocol=https;AccountName=newbreakinstorage;AccountKey=kXz3auR20UJWRGmcnTE+KUeCRiFblk2dr0TcqMaBrehV8tQ7l3fpWIQtA4noLb0KEVRAeaOP/SKRcqU0g/JtbQ==;EndpointSuffix=core.windows.net;')
    
    task = {'PartitionKey': 'Surveyor', 'RowKey': Breakin_id, 'BreakinID': Breakin_id}
    print(task)
    table_service.insert_or_replace_entity('BreakinOutput', task)

def subsample(im,factor):
    a_1 = im.shape[0]//factor
    a_2 = im.shape[1]//factor
    im_new = np.zeros((a_1,a_2),dtype = int)
    for i in range(0,a_1):
        for j in range(0,a_2):
            im_new[i,j]= im[i*factor,j*factor]
    return im_new

def filesize_reducer(test_image_path):
    #starttime  = time.time()
    #os.chdir(insidefolder_path)
    #file_size_list = [os.path.getsize(f)/1048576 for f in os.listdir()]
    #files_list = [f for f in os.listdir() if os.path.isfile(f)]
    #print("Total files in this list are :",len(os.listdir()))
    #count = 0
    for file in os.listdir(test_image_path):
        image = cv2.imread(os.path.join(test_image_path,file))
        #print("Read file Successfully...")
        image_s1 = subsample(image[:,:,0],2)
        image_s2 = subsample(image[:,:,1],2)
        image_s3 = subsample(image[:,:,2],2)
        subsampled_image = np.stack((image_s1,image_s2,image_s3),axis=2)
        #print("Subsampled Image Successfully...")
        im = Image.fromarray(subsampled_image.astype(np.uint8))
        #print("Deleting file :",file)
        os.remove(os.path.join(test_image_path,file))
        #count = count + 1
        b, g, r = im.split()
        im = Image.merge("RGB", (r, g, b))
        im.save(os.path.join(test_image_path,file))
        #im.save("New_"+file)
        #print("Saved ",count," file(s) successfully...")
    #new_file_size_list = [os.path.getsize(os.path.join(test_image_path,f))/1048576 for f in os.listdir()]
    #print("File Size Reduced Successfully with reduced size:",np.sum(new_file_size_list))
    #endtime = time.time()
    #print("Total Time Taken to Reduce the File Size in seconds :",endtime-starttime)
    #print("Total Time Taken to Reduce the File Size in minutes :",(endtime-starttime)/60)




#--------------- Selective Images Changes1 Starts here ---------------#

def Get_List_of_inputs_For_a_BreakinID(Breakin_id):
    connect_str = "DefaultEndpointsProtocol=https;AccountName=newbreakinstorage;AccountKey=0j8YZyoNU8Ku0kPjhQSmCZABSWMsOncuSaBcSeoGgdbVtJijs5Eb8VPoBDLdfRMSDiKItdtyEnKDsBAGxXju4w==;EndpointSuffix=core.windows.net"
    container_name="breakininput"
    
    blob_service = BlobServiceClient.from_connection_string(conn_str = connect_str, container_name = container_name)
    
    container_client = blob_service.get_container_client(container_name)
    
    blob_list = container_client.list_blobs(name_starts_with=Breakin_id)
    
    names = []
    
    for blob in blob_list:
        print("\t Blob name: " + blob.name)
        names.append(blob.name)
    
    return names

#--------------- Selective Images Changes1 Endss here ---------------#

def run(json_string):
    start_time = time.time()
    print("-----Started Processing----")
    print(json_string)
    print("type :", type(json_string))
    json_string = json.loads(json_string)
    print("type2 :", type(json_string))
    results={}
    try:
        print("I AM INSIDE TRY")
        #read the key claim_no from the JSON claim_no
        a = json_string
        for key, value in a.items():
            print("key :", key)
            print("value :", value)
            if key == "data":
                for key1, value1 in value.items():
                        print("key :", key1)
                        print("value :", value1)
                        file_name = value1
            else:
                break
    
        file_name = str(file_name)
        
        #--------------- Selective Images Changes2 Starts here ---------------#
        #--------------- Gettign Breakin ID from The Input File ---------------#
        global Breakin_id
        Breakin_id = file_name.split("_")[0]
        all_files = Get_List_of_inputs_For_a_BreakinID(Breakin_id)
        #--------------- MetData Changes1 Starts here ---------------#
        all_files = [ x for x in all_files if ".zip" in x ]
        print("all_files 1------------------------",all_files)
        #--------------- MetData Changes1 Ends here ---------------#
        
        # Remove Extension from the file Name
        all_files = [os.path.splitext(x)[0] for x in all_files]
        print("all_files 2------------------------",all_files)
        
        all_files.sort(reverse=True)
        print("all_files 3------------------------",all_files)
        
        
        file_name = all_files[0]
        
        #--------------- Selective Images Changes2 Ends here ---------------#
        
        global claim_no1        
       
        claim_no1 = file_name + "/"
        
        
        path_label_template = "pascal_label_map_modelName.pbtxt"
    
        claim_no1 = str(claim_no1)
    
        test_image_path1 = file_name
        #print('test_image_path1======', test_image_path1)
        
        folder_directory = os.getcwd() + '/' + str(file_name) + '/'
        os.mkdir(folder_directory)
        
        # copy_from_blob(file_name)
        #print("name of file which is copied", file_name)
        
        #--------------- Selective Images Changes3 Starts here ---------------#
        #--------------- Downloading all files starts with Breakin ID and unzip into folder ---------------#
        for i in all_files:
            copy_from_blob(i)
            i = i + ".zip"
            with zipfile.ZipFile(i, 'r') as zipObj:
                zipObj.extractall(folder_directory)
        
        #--------------- Selective Images Changes3 Ends here ---------------#
        
        
        #--------------- Additional Images Fix Changes1 Starts here ---------------#
        
        req_img_list = ['SIDE','WINDSHIELD_Customer','CHASSIS_NO_Customer','RPM_READING_Customer','ODOMETER_READING_Customer']
        
        for filename in os.listdir(folder_directory):
            if any(ele in filename for ele in req_img_list):
                pass
            else:
                os.remove(folder_directory + '/' + filename)
                print(filename)
        

        #--------------- Additional Images Fix Changes1 Ends here ---------------#
        
        
        
        
        #--------------- Selective Images Changes4 Starts here ---------------#
        
        # file_name = file_name + ".zip"
        
        #--------------- Selective Images Changes4 Ends here ---------------#
        
        
        
        hitlim = 0.4
        
        
        #--------------- Selective Images Changes5 Starts here ---------------#
        
        
        # #print('file_name before unzip======', file_name)
        # with zipfile.ZipFile(file_name, 'r') as zipObj:
        #     zipObj.extractall(folder_directory)
        # #path = os.getcwd()
        
        # zip  = zipfile.ZipFile(file_name)
        # img_list = zip.namelist()
        # global Breakin_id
        # Breakin_id = img_list[0][0:7]
        # print('======breakin_ID is=====', Breakin_id)
        
        #--------------- Selective Images Changes5 Ends here ---------------#
        
        
        print("Checking the Size of Zip file...")
        file_size = os.path.getsize(file_name)/1048576 #file_size is in mb    
        print("Total Size of Zip Folder :",np.round(file_size,2))
    
    
    
        #--------------- Selective Images Changes6 Starts here ---------------#
    
        # os.remove(file_name)
        
        #--------------- Selective Images Changes6 Ends here ---------------#
        
        
        
        
        #--------------- Selective Images Changes7 Starts here ---------------#
        #--------------- Deleting all files starts with Breakin ID ---------------#
        
        for f in all_files:
            os.remove(f + ".zip")
        
        #--------------- Selective Images Changes7 Ends here ---------------#
        
        # Check_existing_breakin(Breakin_id)
        insert_entity("", "", Breakin_id)
        update_table_status("Processing", "", Breakin_id)
        
        test_image_path = op.join(os.getcwd(), test_image_path1)
        
        if file_size > 50:
            print("Reducing File Size...")
            filesize_reducer(test_image_path)
        else:
            print("File Size is within the range...")

        
        car_final, Final_NBlur1, Output, Blur1, car_not_found, NBlur1, Output_Blur1,odo_check,RPM_check,Engine_On_and_Off_check,Chassis_check = images_to_process_2(test_image_path,Breakin_id)
        
        global final_recapture_flag
        
        final_recapture_flag = combine_flag(RPM_check,Chassis_check,odo_check)
        print("final_recapture_flag==============:",final_recapture_flag)
        if (odo_check == 1  and Chassis_check == 1):
        # if (odo_check == 1 and RPM_check == 1 and Chassis_check == 1):
        # if (Chassis_check == 1):
            
            if len(NBlur1) != 0:        
                data_part = tf_od_pred("part", Final_NBlur1)
                #print('------data_part----', data_part)
                
                if len(data_part) > 0:
                    data_dent = tf_od_pred("dent", Final_NBlur1)
                    data_dent = data_dent[data_dent['class'] == 'Dent']
                    data_scratch = tf_od_pred("scratch", Final_NBlur1)
                    data_scratch = data_scratch[data_scratch['class'] == 'Scratch']
                    data_crack = tf_od_pred("crack", Final_NBlur1)
                    data_angle = tf_od_pred("meterLR", Final_NBlur1)
                    data_FP = tf_od_pred("fpdamages", Final_NBlur1)
                    
                    #------------------Saving intrim model output starts here -----------------#
                    #========Car Final======
                    car_final.to_csv(str(Breakin_id) + '_car_final_' + str(present_time1) + '.csv')
                    print("car_final=============== :", car_final)
                    copy_to_blob_custom(str(Breakin_id) + '_car_final_' + str(present_time1),'summaryanalysis')     
                    os.remove(str(Breakin_id) + '_car_final_' + str(present_time1) + '.csv')
                    
                    #========Data Part======
                    data_part.to_csv(str(Breakin_id) + '_data_part_' + str(present_time1) + '.csv')
                    print("data_part=============== :", data_part)
                    copy_to_blob_custom(str(Breakin_id) + '_data_part_' + str(present_time1),'summaryanalysis')     
                    os.remove(str(Breakin_id) + '_data_part_' + str(present_time1) + '.csv')
                    
                    #========Data Dent======
                    data_dent.to_csv(str(Breakin_id) + '_data_dent_' + str(present_time1) + '.csv')
                    print("data_dent=============== :", data_dent)
                    copy_to_blob_custom(str(Breakin_id) + '_data_dent_' + str(present_time1),'summaryanalysis')     
                    os.remove(str(Breakin_id) + '_data_dent_' + str(present_time1) + '.csv')
                    
                    #========Data Scratch======
                    data_scratch.to_csv(str(Breakin_id) + '_data_scratch_' + str(present_time1) + '.csv')
                    print("data_scratch=============== :", data_scratch)
                    copy_to_blob_custom(str(Breakin_id) + '_data_scratch_' + str(present_time1),'summaryanalysis')     
                    os.remove(str(Breakin_id) + '_data_scratch_' + str(present_time1) + '.csv')
                    
                    #========Data Crack======
                    data_crack.to_csv(str(Breakin_id) + '_data_crack_' + str(present_time1) + '.csv')
                    print("data_crack=============== :", data_crack)
                    copy_to_blob_custom(str(Breakin_id) + '_data_crack_' + str(present_time1),'summaryanalysis')     
                    os.remove(str(Breakin_id) + '_data_crack_' + str(present_time1) + '.csv')
                    
                    #========Data Angle======
                    data_angle.to_csv(str(Breakin_id) + '_data_angle_' + str(present_time1) + '.csv')
                    print("data_angle=============== :", data_angle)
                    copy_to_blob_custom(str(Breakin_id) + '_data_angle_' + str(present_time1),'summaryanalysis')     
                    os.remove(str(Breakin_id) + '_data_angle_' + str(present_time1) + '.csv')
                    
                    #------------------Saving intrim model output ends here -----------------#
                    
                    #print("***********Prior to call to prepare_output")
                    final_output, l_angle_final, car_angle = prepare_output(car_final, Output, data_dent, data_scratch, data_crack, data_FP, car_not_found, Blur1, data_angle, data_part) #, data_np)
                    
                    print('--------final_output after prepare output----', final_output)
                    print('--------l_angle_final after prepare output----', l_angle_final)
                    print('--------car_angle after prepare output----', car_angle)
                    
                    if Output_Blur1 is not None:
                        final_output = final_output.append(Output_Blur1, ignore_index=True)
                else:
                    car_angle = 0
                    final_output = pd.DataFrame(columns = ['id', 'Damage_Type', "score_y", "class_x", "score_x", "Side", "score", "part_damage_dist"], dtype=object)
                    final_output.loc[0] = ['Summary', 'No Car Image Found', '', '', '', '', '', '']
                    #Output_np = pd.DataFrame(columns= ['id', 'Damage_Type', 'class_x', 'score_x', 'Side', 'score', 'part_damage_dist'])
                    
                    
                    #print("preparing final output11111111", final_output)
            else:
                car_angle = 0
                final_output = (Output_Blur1)
                final_output.loc[0] = ['Summary', 'All images are blur (or) No Car Image Found', '', '', '', '', '', '']
           
            print("final output: ", final_output)
            
            final_output = final_output[final_output['Damage_Type'].notnull()]
            final_output  = final_output.drop_duplicates(subset=['id', 'Damage_Type', 'score_y', 'class_x', 'score_x',	'Side',	'score', 'part_damage_dist'], inplace=False)
           
            print("final output list of variables: ", list(final_output))
            
            # summary1 = final_output[final_output['Damage_Type'].isin(['No Car Image Found'])]
            # summary2 = final_output[final_output['Damage_Type'].isin(['do not have min no. of angles'])]
            summary2 = final_output[final_output['Damage_Type'].isin(['Less than 8 images have car/few images are blur','do not have min no. of angles'])]
            print(summary2)
            if len(summary2) == 0:
                summary1 = final_output[final_output['Damage_Type'].isin(['No Car Image Found'])]
                print(summary2)
                blur_images = list(Output_Blur1['id'])
                car_not_found = list(car_not_found)
                
                no_car_and_blur = common(car_not_found, blur_images)
                print("length of summary1 --------------",len(summary1))
                print("length of summary2 --------------",len(summary2))
            # if len(summary2) == 0:
                if len(summary1) == 0:
                    final_output_new, remain_cases = Repair_Replace(final_output, car_angle)
                    
                
                else:
                    final_output_new = pd.DataFrame(columns = ['id', 'Damage_Type', 'score_y' ,'class_x', 'score_x', 'score', 'part_damage_dist', 'Side', 'Type', 'Final_Flag'], dtype=object)
                    final_output_new.loc[len(final_output_new)+1] = ['Summary', 'No Car Image Found', '', '', '', '', '', '', '', '']
                    remain_cases = pd.DataFrame(columns = ['id', 'Damage_Type', 'score_y' ,'class_x', 'score_x', 'Side', 'score', 'part_damage_dist'], dtype=object)
                    
                    
                remain_cases = Final_Result1(remain_cases)
                
                
                if len(remain_cases) != 0:
                    for h in no_car_and_blur:
                            
                        aa = remain_cases[(remain_cases['id'] == h)].index
                        
                        remain_cases.drop(aa[0], inplace=True)
                else:
                    remain_cases = remain_cases
                
                final_output = final_output_new  
                
                
                final_output = final_output.append(remain_cases, ignore_index=True)
                print("final output-==============================================",final_output)
                #convert Bro_ws to broken windshield and etc........
                final_output = damage_update(final_output)
                
                final_count1 = final_count(final_output)
                print("final output 2222222222222222222222222222222222222222222222 ",final_count1)
                #final_output.to_csv("temp/finalop_after_250_1_" + str(Breakin_id) + ".csv")
                #final_output.to_csv('After_final_count_original.csv', index =False)
                final_count1 = final_count1[['id', 'Damage_Type', 'class_x', 'Side']]
         
                final_count1 = final_count1.rename(columns={"id": "Image_Name"})
                #final_count1 = final_count1.rename(columns={"id": "Image_Name"})
                
                final_count1['BREAKIN_ID'] = Breakin_id
                
                final_count1.loc[len(final_count1)+1] = ['code', str(final_recapture_flag), '', '', Breakin_id]
            
            else:
                final_count1 = pd.DataFrame(columns = ['Image_Name', 'Damage_Type', 'class_x', 'Side', 'BREAKIN_ID'], dtype=object)
                final_count1.loc[0] = ['Note', 'Proper side is missing', '', '', Breakin_id]
                final_count1.loc[1] = ['code', str(final_recapture_flag), '', '', Breakin_id]
                final_count1.loc[2] = ['Summary', 'Less than 8 images have car/few images are blur', '', '', Breakin_id]
        else:
            
            final_count1 = pd.DataFrame(columns = ['Image_Name', 'Damage_Type', 'class_x', 'Side', 'BREAKIN_ID'], dtype=object)
            final_count1.loc[0] = ['Note', 'Chassis Number or Odometer not found', '', '', Breakin_id]
            # final_count1.loc[0] = ['Note', 'Chassis Number or RPMmeter or Odometer not found', '', '', Breakin_id]
            final_count1.loc[1] = ['code', str(final_recapture_flag), '', '', Breakin_id]
            final_count1.loc[2] = ['Summary', 'Less than 8 images have car/few images are blur', '', '', Breakin_id]
        
        final_count1  = final_count1.drop_duplicates(subset=['Image_Name', 'Damage_Type', 'class_x', 'Side', 'BREAKIN_ID'], inplace=False)
        
        final_count1 = final_count1[['Image_Name', 'Damage_Type', 'class_x', 'Side', 'BREAKIN_ID']]
        
        #---------------------------Saving results before sending to IT---------------------#
             
        final_count1.to_csv(str(Breakin_id) + '_final_count1_Before_Changing_results_' + str(present_time1) +'.csv', index =False)
        copy_to_blob_custom(str(Breakin_id) + '_final_count1_Before_Changing_results_' + str(present_time1),'prechangeresults')
        os.remove(str(Breakin_id) + '_final_count1_Before_Changing_results_' + str(present_time1) +'.csv')
        
        
        #---------------------------Saving results before sending to IT---------------------#
        
        
        #--------------------- Changes in Final Count Starts Here ---------------------#
        
        #Tackle1
        #Delete Blank Image_Name entries
        final_count1 = final_count1.dropna(axis=0, subset=['Image_Name'])
        
        #Delete entries like Chassis Not Found----
        final_count1.loc[final_count1['Image_Name'].str.contains('CHASSIS_NO|ODOMETER_READING|WINDSHIELD|RPM_READING|VEHICAL_REGISTRATION_NO'),'Final_Flag_1'] = 1
        
        final_count1['Final_Flag_1'].fillna(0, inplace=True)
        
        #Tackle2
        final_count2 = final_count1.assign(Final_Flag=final_count1.apply(Flag_creation_Last, axis=1))
        final_count3 = final_count2[final_count2['Final_Flag'] == 0]
        final_count4 = final_count3.drop(['Final_Flag','Final_Flag_1'], axis=1)
        final_count1 = final_count4.copy()
        
        
        
        #--------------------- Changes in Final Count Ends Here ---------------------#
        
        file1 = final_count1[final_count1.Image_Name == 'Summary'] 
        print("final summary---------------------------")
        print(file1)
        print(file1['Damage_Type'])
        print("final summary---------------------------")
        file2 = final_count1[final_count1.Image_Name != 'Summary']   
        frames = [file2, file1] 
        final_count1 = pd.concat(frames, ignore_index=True)
        
        
        #---------------------------Saving results before sending to IT---------------------#
        
        final_count1.to_csv(str(Breakin_id) + '_final_count1_After_Changing_results_' + str(present_time1) +'.csv', index =False)
        
        copy_to_blob_custom(str(Breakin_id) + '_final_count1_After_Changing_results_' + str(present_time1),'afterchangeresults')
        
        os.remove(str(Breakin_id) + '_final_count1_After_Changing_results_' + str(present_time1) +'.csv')
        
        #---------------------------Saving results before sending to IT---------------------#
        
        
        if (final_count1 is not None):
            #print("final output in JSON: ", final_count1.to_json(orient='records'))
            update_table_status("Completed", final_count1.to_json(orient='records'), Breakin_id)
        else:
            update_table_status("CompletedWithError", "No side images found", Breakin_id)
        
    
        shutil.rmtree(test_image_path) 
        end_time = time.time()
        time_taken = end_time - start_time
        time_df = pd.DataFrame(
                    {"Breakin_id" : [Breakin_id],
                     "time_taken" : [time_taken]})
        
        time_df.to_csv("time_" + str(Breakin_id) + ".csv" , index =False)
        check_existing("time_" + str(Breakin_id))
        copy_to_blob("time_" + str(Breakin_id))
        os.remove("time_" + str(Breakin_id) + ".csv")
        
        
        print("--- %s seconds ---" % (end_time - start_time))
        print("--- %s minutes ---" % ((end_time - start_time)/60))
            #Save files to blob, send it to somewhere
        #print(final_output_new)    
        print("HURREEEEEEEEY IT IS DONEEEEEEEEEEEEEEEEE")
    except Exception as e:
        error = str(e)
        update_table_status("CompletedWithError", "", "")
        results['Output'] = Breakin_id
        results['Status'] = error
        update_table_status("CompletedWithError", "", Breakin_id)
    
        
    print(results)
    return results
    
    #return results

init()

run('{"data": {"claim_no": "'+ sys.argv[1].replace(".zip","") +'"}}')