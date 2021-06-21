from __future__ import print_function, division
from itertools import filterfalse
from re import A
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import pydicom as dicom
import shutil
import cv2
import numpy as np
import os
import psutil
import tkinter as tk
import tkinter.font as tkFont
import random
from tkinter import ttk
import mod.GUI.GUI as GUI
import mod.diagnostic as diag
import mod.clinical_model as clinical
from mod.image_model import image_model
from mod.GUI.GUI_audio import gui_audio
from mod.NLP.NLP import NLP
from mod.NLP.speech_txt_conv import sph_txt, txt_sph, recorder

# un-comment to show all of pandas dataframe
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)

# un-comment to show all of numpy array
#np.set_printoptions(threshold=sys.maxsize)

useDefaults = GUI.indexPage.useDefaults

use_audio = False

if useDefaults:
    # SPECIFY VARIABLES HERE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    save_fit = False
    load_fit = False
    model_save_loc = "D:\Cancer_Project\Cancer_ML\HNSCC-HN1\saved_model (CNN)"

    main_data = "D:\Cancer_Project\Cancer_ML\data\HNSCC-HN1\Copy of HEAD-NECK-RADIOMICS-HN1 Clinical data updated July 2020 (adjusted chemotherapy var).csv"
    sec_data = ""
    test_file = ""

    # list with strings or a single string may be inputted
    target_variables = 'chemotherapy_given'

    # if true, converted images will be in png format instead of jpg
    png = False

    # folder containing Cancer Imagery
    load_dir = "D:\Cancer_Project\Cancer Imagery\HEAD-NECK-RADIOMICS-HN1\HEAD-NECK-RADIOMICS-HN1"

    # directory to save data such as converted images
    save_dir = "D:\\Cancer_Project\\converted_img"

    # directory to save imagery array
    img_array_save = "D:\Cancer_Project\converted_img"

    # if true, numpy image array will be searched for in img_array_save
    load_numpy_img = True

    # if true, attempt will be made to convert dicom files to npy
    convert_imgs = False

    #if true, converted dicom images will be deleted after use
    del_converted_imgs = False

    # if true, image model will be ran instead of clinical only model
    run_img_model = False

    # if true, two data files will be expected for input
    two_datasets = False

    # if true, an additional file will be expected for testing
    use_additional_test_file = False

    # where image id is located in image names (start,end)
    # only applies if using image model
    img_id_name_loc = (9, 12)

    # Column of IDs in dataset. Acceptable values include "index" or a column name.
    ID_dataset_col = "id"

    # tuple with dimension of imagery. All images must equal this dimension
    img_dimensions = (512, 512)

    # if true, every column in data will be inputted for target variable
    target_all = False

    # save location for data/graphs
    data_save_loc = "D:\\Cancer_Project\\Cancer_ML\\result_graphs"

    # if true, graphs will be shown after training model
    show_figs = True

    # if true, graphs will be saved after training model
    save_figs = False

    # if true, convert dicom directly to numpy. Otherwise, convert to jpg or png first in save_dir
    dcmDirect = True

    # number of epochs in model
    num_epochs = 20

    # if true, CNN will be used
    useCNN = False

    # if true, diagnosis model will run
    diagModel = False

    # END VARIABLES - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
else:

    boolList = GUI.boolList

    # convert every element in boolList to a proper boolean
    [bool(b) for b in boolList]

    dictTxt = dict(zip(GUI.varList_txt, GUI.txtEntry_list))
    dictBool = dict(zip(GUI.varList_bool, boolList))

    save_fit = dictBool["save_fit "]
    model_save_loc = dictTxt["model_save_loc "]

    main_data = dictTxt["main_data "]
    sec_data = dictTxt["sec_data "]
    test_file = dictTxt["test_file "]

    # list with strings or a single string may be inputted
    # check if string is list. Find returns -1 if value cannot be found
    if dictTxt["target_variables "].find("[") != -1 and dictTxt["target_variables "].find(",") != -1:
        target_variables = list(dictTxt["target_variables "][1:-1].split(","))

        # remove excess quotes
        target_variables = ([v.strip("'") for v in target_variables])
        target_variables = ([v.replace("'",'') for v in target_variables])
    else:
        target_variables = dictTxt["target_variables "]

    # if true, converted images will be in png format instead of jpg
    png = dictBool["png "]

    # folder containing Cancer Imagery
    load_dir = dictTxt["load_dir "]

    # directory to save data such as converted images
    save_dir = dictTxt["save_dir "]

    # directory to save imagery array
    img_array_save = dictTxt["img_array_save "]

    # if true, numpy image array will be searched for in img_array_save
    load_numpy_img = dictBool["load_numpy_img "]

    # if true, attempt will be made to convert dicom files to jpg or png
    convert_imgs = dictBool["convert_imgs "]

    #if true, converted dicom images will be deleted after use
    del_converted_imgs = dictBool["del_converted_imgs "]

    # if true, image model will be ran instead of clinical only model
    run_img_model = dictBool["run_img_model "]

    # if true, two data files will be expected for input
    two_datasets = dictBool["two_datasets "]

    # if true, an additional file will be expected for testing
    use_additional_test_file = dictBool["use_additional_test_file "]

    # where image id is located in image names (start,end)
    # only applies if using image model
    img_id_name_loc = dictTxt["img_id_name_loc "]

    # Column of IDs in dataset. Acceptable values include "index" or a column name.
    ID_dataset_col = dictTxt["ID_dataset_col "]

    # tuple with dimension of imagery. All images must equal this dimension
    img_dimensions = dictTxt["img_dimensions "]

    # if true, every column in data will be inputted for target variable
    target_all = dictBool["target_all "]

    # save location for data/graphs
    data_save_loc = dictTxt["data_save_loc "]

    # if true, graphs will be shown after training model
    show_figs = dictBool["show_figs "]

    # if true, graphs will be saved after training model
    save_figs = dictBool["save_figs "]

    # if true, convert dicom to standard format before put into numpy
    dcmDirect = dictBool["dcmDirect"]

    # number of epochs in model
    num_epochs = int(dictTxt["num_epochs "])

    useCNN = dictBool["useCNN "]

    diagModel = dictBool["diagModel "]

if diagModel:
    diag = diag.diagnostic(main_data,target_variables,20)
    diag.model()

mainPath = main_data

if use_audio:
    audio_ui = gui_audio()
    audio_ui.run()

    engine = NLP()
    engine.run()

    print(engine.variable)

def cleanData(pd_dataset):
    df = pd_dataset.dropna()
    return df

codeDict = {}
def encodeText(dataset):
    global codeDict

    if str(type(dataset)) == "<class 'str'>":
        dataset = pd.read_csv(dataset, low_memory=False)

    dataset = cleanData(dataset)

    dShape = dataset.shape
    a1 = dShape[0]
    a2 = dShape[1]

    if a1 >= a2:
        longestAxis = a1
        shortestAxis = a2
    else:
        longestAxis = a2
        shortestAxis = a1

    wordList = []
    for i in range(longestAxis): 
        for n in range(shortestAxis): 
            if longestAxis == a1: 
                data = dataset.iloc[i, n]
            else: 
                data = dataset.iloc[n, i]

            if str(type(data)) == "<class 'str'>":

                # list of chars to be removed from data
                char_blocked = [' ', '.', '/', '-', '_', '>', '+', ',', ')', '(', '*',
                                '=', '?', ':', '[', ']', '#', '!']

                for char in char_blocked: 
                    if char in data: 
                        data = data.replace(char, '')

                if longestAxis == a1:
                    dataset.iloc[i, n] = data
                else: 
                    dataset.iloc[n, i] = data

                wordList.append(data)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(wordList)
    codeDict = tokenizer.word_index

    print(codeDict)

    for i in range(longestAxis):
        for n in range(shortestAxis): 
            if longestAxis == a1: 
                data = dataset.iloc[i, n]
            else: 
                data = dataset.iloc[n, i]
            
            if str(type(data)) == "<class 'str'>":
                data = data.lower()
                data = int(codeDict[data])

            if longestAxis == a1: 
                dataset.iloc[i, n] = data
            else: 
                dataset.iloc[n, i] = data

    # convert all cols to numeric vals
    dataset = dataset.astype('int64')
                
    return dataset

main_data = encodeText(main_data)

col = None
# function for determining if target variable(s) are binary val
# returns bool if single var 
# returns list of bools in corresponding order to target variables list if multiple vars   
def isBinary(target_var): 
    global col 

    orgPD = pd.read_csv(mainPath)
    orgPD = orgPD.dropna()

    # check if param is a list of multiple vars 
    if str(type(target_var)) == "<class 'list'>" and len(target_var) > 1:

        for vars in target_var: 

            # initialize list to hold bools 
            areBinary = []
        
            col = list(orgPD[vars])

            # remove duplicates 
            col = list(set(col))

            # check if data is numerical 
            for vals in col: 
                if str(type(vals)) == "<class 'int'>" or str(type(vals)) == "<class 'float'>": 
                    numeric = True
                else: 
                    numeric = False 

            if not numeric: 

                if len(col) == 2: 
                    isBinary = True
                else: 
                    isBinary = False 

                areBinary.append(isBinary)
            else: 
                areBinary = False

        isBinary = areBinary 

    else: 

        col = list(orgPD[target_var])

        # remove duplicates 
        col = list(set(col))

        # check if original data is numerical
        for vals in col: 
            if str(type(vals)) == "<class 'int'>" or str(type(vals)) == "<class 'float'>": 
                numeric = True
            else: 
                numeric = False 
        
        if not numeric: 
            if len(col) == 2: 
                isBinary = True
            else: 
                isBinary = False 

        else: 
            isBinary = False

    return isBinary

isBinary = isBinary(target_variables)

# make dictionary with definitions for only target var 
convCol = main_data.loc[:,target_variables]
if str(type(target_variables)) == "<class 'list'>" and len(target_variables) > 1: 
    valList = []
    for cols in convCol: 
        for vals in list(cols): 
            valList.append(vals)

    valList = list(set(valList))

    smNum = min(valList)
    lgNum = max(valList)

    valList[valList.index(smNum)] = 0
    valList[valList.index(lgNum)] = 1

    orgPD = pd.read_csv(mainPath)
    orgPD = orgPD.dropna()

    orgList = []
    for cols in orgPD.loc[:,target_variables]: 
        for vals in list(cols):
            orgList.append(vals)

    orgList = list(set(orgList))
    
    targetDict = dict(zip(valList,orgList))

else: 

    valList = []
    for vals in list(convCol): 
        valList.append(vals)

    valList = list(set(valList))

    smNum = min(valList)
    lgNum = max(valList)

    valList[valList.index(smNum)] = 0
    valList[valList.index(lgNum)] = 1

    orgPD = pd.read_csv(mainPath)
    orgPD = orgPD.dropna()

    orgList = []
    for vals in orgPD.loc[:,target_variables]:  
        orgList.append(vals)
    
    orgList = list(set(orgList))

    targetDict = dict(zip(valList,orgList))

# function to decode post-training vals into text
# only use with binary values
# function rounds vals to convert  
def decode(iterable,codeDict): 
    
    if str(type(iterable)) == "<class 'list'>": 
        iterable = np.array(iterable)

    initialShape = iterable.shape
    
    iterable = iterable.flatten()

    iterable = np.around(iterable,decimals=0)

    dictKeys = list(codeDict.keys())
    dictVals = list(codeDict.values())

    # determine type of vals
    # initialize text bool as false 
    textKeys = False 
    for keys in dictKeys: 
        if str(type(keys)) == "<class 'str'>": 
            textKeys = True

    if not textKeys: 
        i = 0 
        for keys in dictKeys: 
            keys = round(keys,0)
            dictKeys[i] = keys
            i = i + 1 
    else: 
        i = 0 
        for vals in dictVals:
            try:
                vals = round(vals,0)
                dictVals[i] = vals
            except:
                i = i + 1

    roundedDict = dict(zip(dictKeys,dictVals))

    def target_dict(): 
        colData = main_data.loc[:,target_variables]
        try: 
            for cols in list(colData.columns): 
                col = colData[cols].tolist()
                col = list(set(col))
        except: 
            col = colData.tolist()
            col = list(set(col))

    if isBinary: 
        target_dict()
    
    convIt = []
    for vals in iterable: 
        tran = roundedDict[vals]
        convIt.append(tran)

    convIt = np.array(convIt)

    # make array back into initial shape
    convIt = np.reshape(convIt,initialShape)

    return convIt

# function that returns percentage accuracy from rounded values
def percentageAccuracy(iterable1,iterable2):
    
    def roundList(iterable):

        if str(type(iterable)) == "<class 'tensorflow.python.framework.ops.EagerTensor'>":
            iterable = iterable.numpy()
        roundVals = []
        if int(iterable.ndim) == 1:
            for i in iterable: 
                i = round(i,0)
                roundVals.append(i)
        
        elif int(iterable.ndim) == 2: 
            for arr in iterable: 
                for i in arr: 
                    i = round(i,0)
                    roundVals.append(i)

        elif int(iterable.ndim) == 3:
            for dim in iterable:
                for arr in dim:
                    for i in arr:
                        i = round(i,0)
                        roundVals.append(i)

        elif int(iterable.ndim) == 4:
            for d in iterable:
                for dim in d:
                    for arr in dim:
                        for i in arr:
                            i = round(i,0)
                            roundVals.append(i)

        else:
            print("Too many dimensions--ERROR")

        return roundVals

    rounded1 = roundList(iterable1)
    rounded2 = roundList(iterable2)

    # remove negative zeros from lists
    i = 0
    for vals in rounded1:
        if int(vals) == -0 or int(vals) == 0:
            vals = abs(vals)
            rounded1[i] = vals

        i = i + 1

    i = 0
    for vals in rounded2:
        if int(vals) == -0 or int(vals) == 0:
            vals = abs(vals)
            rounded2[i] = vals

        i = i + 1

    numCorrect = len([i for i, j in zip(rounded1, rounded2) if i == j])

    listLen = len(rounded1)

    percentCorr = numCorrect/listLen
    percentCorr = percentCorr * 100

    percentCorr = round(percentCorr,2)

    return percentCorr

def GUI_varConnector(dataset1, dataset2):

    if str(type(dataset1)) == "<class 'str'>":
        dataset1 = pd.read_csv(dataset1)

    if str(type(dataset2)) == "<class 'str'>":
        dataset2 = pd.read_csv(dataset2)

    vars1 = list(dataset1.columns)
    vars2 = list(dataset2.columns)

    vars1.remove(ID_dataset_col)
    vars2.remove(ID_dataset_col)

    for element in target_variables:
        if element in vars1:
            vars1.remove(element)
        if element in vars2:
            vars2.remove(element)

    # list of colors for buttons to choose from
    colors = ["red", "blue", "purple", "orange", "green", "gray",
              "gainsboro", "dark salmon", "LemonChiffon2", "ivory3",
              "SteelBlue1", "DarkOliveGreen3", "gold2", "plum1"]

    window = tk.Tk()

    window.title("Variable Connector")
    window.iconbitmap("D:\Cancer_Project\Cancer_ML\cancer_icon.ico")

    main_frame = tk.Frame(window)
    main_frame.pack(fill=tk.BOTH,expand=1)

    canvas = tk.Canvas(main_frame)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    # Add a scrollbars to the canvas
    scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    scrollbar_x = ttk.Scrollbar(main_frame,orient=tk.HORIZONTAL, command=canvas.xview)
    scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

    # Configure the canvas
    canvas.configure(xscrollcommand=scrollbar_x.set)
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    second_frame = tk.Frame(canvas)
    canvas.create_window((0,0), window=second_frame, anchor="nw")

    buttonFont = tkFont.Font(family="Georgia", size=20)
    font = tkFont.Font(family="Georgia",size=25)
    title = tk.Label(text="Select matching variables", font=font, fg="#0352fc")
    title.place(relx=0.2,rely=0)

    button = None

    pressedVars = []
    buttonList = []

    def makeButtons(var_name, x, y):
        var = var_name

        def trackVars():
            pressedVars.append(var)
            button.config(bg=random.choice(colors))

        button = tk.Button(master=second_frame,text=var_name, fg="white", bg="black", width=30, height=1,
                           command=trackVars,font=buttonFont)
        button.grid(column=x,row=y,padx=105,pady=50)
        buttonList.append(button)

    y = 1
    for var in vars1:
        makeButtons(var, 10, y)
        y = y + 10

    y = 1
    for var2 in vars2:
        makeButtons(var2, 20, y)
        y = y + 10

    exitButton = tk.Button(master=second_frame,text="Done",fg="white",bg="orange",width=30,height=3,
                           command=window.destroy)
    exitButton.grid(row=1,column=100)

    window.mainloop()

    # function used to convert list to dictionary
    def Convert(lst):
        res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
        return res_dct

    pressedVars_dict = Convert(pressedVars)
    return pressedVars_dict

if two_datasets == True:
    varMatches = GUI_varConnector(main_data,sec_data)
    print(varMatches)

def collect_img_dirs(data_folder):
    img_directories = []

    for root, dirs, files, in os.walk(data_folder):
        for name in files:
            dir = os.path.join(root,name)
            img_directories.append(dir)

    return img_directories

if convert_imgs == True:
    load_dirs = collect_img_dirs(load_dir)

def convert_img(png_boolean,dirs_list,save_path):
    png = png_boolean

    print("starting image conversion process")
    num_converted_img = 0
    for image in dirs_list:

        # filter out incompatible images
        if os.path.basename(image) != "1-1.dcm":
            ds = dicom.dcmread(image)
            pixel_array_numpy = ds.pixel_array

            if png == False:
                image = image.replace(".dcm",".jpg")
            elif png == True:
                image = image.replace(".dcm",".png")

            cv2.imwrite(os.path.join(save_path,ds.PatientID+"_"+os.path.basename(image)),pixel_array_numpy)

            ## Loading info
            num_imgs = len(dirs_list)
            num_converted_img = num_converted_img + 1
            percentage_done = (num_converted_img/num_imgs) * 100
            print(str(round(percentage_done,2)) + " percent completed")

def convert_npy(dirs_list,save_path):
    print("appending dicom files directly to numpy array")
    img_array = np.array([])
    img_conv = 0
    for f in dirs_list:

        # filter incompatible images
        if os.path.basename(f) != "1-1.dcm":
            ds = dicom.dcmread(f)
            pixel_array_numpy = ds.pixel_array
            id = ds.PatientID

            for s in id:
                if not s.isdigit():
                    id = id.replace(s,'')

            id = int(id[-3:])

            if pixel_array_numpy.shape == img_dimensions:
                pixel_array_numpy = pixel_array_numpy.flatten()
                pixel_array_numpy = np.insert(pixel_array_numpy,len(pixel_array_numpy),id)
                img_array = np.append(img_array,pixel_array_numpy)

        print(psutil.virtual_memory().percent)

        # memory optimization
        if psutil.virtual_memory().percent >= 60:
            break

        ## Loading info
        num_imgs = len(dirs_list)
        img_conv = img_conv + 1
        percentage_done = (img_conv / num_imgs) * 100
        print(str(round(percentage_done, 2)) + " percent completed")

    np.save(os.path.join(save_path, "img_array"), img_array)

if convert_imgs == True and dcmDirect == False:
    convert_img(png, load_dirs,save_dir)
elif convert_imgs == True and load_numpy_img == False and dcmDirect == True:
    convert_npy(load_dirs,save_dir)

def prep_data(data_file_1,data_file_2):
    if str(type(data_file_1)) != "<class 'pandas.core.frame.DataFrame'>":
        file_1 = pd.read_csv(data_file_1)
    else:
        file_1 = data_file_1

    common_ids = []

    if ID_dataset_col != "index":
        file_1 = file_1.set_index(ID_dataset_col)

    ids_1 = file_1.index

    if two_datasets == True:
        if str(type(data_file_2)) != "<class 'pandas.core.frame.DataFrame'>":
            file_2 = pd.read_csv(data_file_2)
        else:
            file_2 = data_file_2

        file_2 = file_2.set_index(ID_dataset_col)
        ids_2 = file_2.index
        # determine the largest dataset to put first in the for statement
        if ids_1.shape[0] > ids_2.shape[0]:
            longest_ids = ids_1.values.tolist()
            shortest_ids = ids_2.values.tolist()
        elif ids_1.shape[0] < ids_2.shape[0]:
            longest_ids = ids_2.values.tolist()
            shortest_ids = ids_1.values.tolist()
        elif ids_1.shape[0] == ids_2.shape[0]:
            longest_ids = ids_1.values.tolist()
            shortest_ids = ids_2.values.tolist()

        for i in longest_ids:
            for z in shortest_ids:
                if int(i) == int(z):
                    common_ids.append(i)

        adapted_1 = file_1.loc[common_ids]
        adapted_2 = file_2.loc[common_ids]
        combined_dataset = adapted_1.join(adapted_2)

        # eliminate duplicate variables
        for i in varMatches.values():
            combined_dataset = combined_dataset.drop(i,axis=1)
        data = combined_dataset
    else:
        data = file_1

    return data

if two_datasets == True:
    main_data = prep_data(main_data,sec_data)
elif two_datasets == False:
    main_data = prep_data(main_data,None)

resultList = []
prediction = []

def feature_selection(pd_dataset,target_vars,num_features):

    # initialize bool as false
    multiple_targets = False

    if str(type(target_vars)) == "<class 'list'>" and len(target_vars) > 1:
        multiple_targets = True

    corr = pd_dataset.corr()

    # get the top features with the highest correlation
    if multiple_targets == False:
        features = list(pd_dataset.corr().abs().nlargest(num_features,target_vars).index)
    else:
        features = []
        for vars in target_vars:
            f = pd_dataset.corr().abs().nlargest(num_features,vars).index
            f = list(f)
            features.append(f)

        features = sum(features,[])

    # get the top correlation values
    if multiple_targets:
        corrVals=[]
        for vars in target_vars:
            c = pd_dataset.corr().abs().nlargest(num_features,vars).values[:,pd_dataset.shape[1]-1]
            c = list(c)
            corrVals.append(c)

        corrVals = sum(corrVals,[])
    else:
        corrVals = list(pd_dataset.corr().abs().nlargest(num_features,target_vars).values[:,pd_dataset.shape[1]-1])

    # make a dictionary out of the two lists
    featureDict = dict(zip(features,corrVals))

    return featureDict

def format_data(data_file, test_file, target_var):

        if str(type(data_file)) == "<class 'pandas.core.frame.DataFrame'>":
            df = data_file
        elif main_data[-4:] == ".csv":
            df = pd.read_csv(data_file)

        if use_additional_test_file == True:
            #Recognizing what variables are in the test data
            input_data = pd.read_csv(test_file)
            input_vars = input_data.columns.tolist()

            #collect data for the variables from main dataset
            dataset = df[input_vars]

            # Append y data for target column into new dataset
            y_data = df[target_var]
            dataset = dataset.assign(target_variables=y_data)
            target_name = str(target_var)
            dataset = dataset.rename(columns={'target_variables':target_name},inplace=True)
        elif use_additional_test_file == False:
            dataset = df

        return dataset

def postTrain(multiple_targets,y_val,X_val,X_test,y_test,model):
    # utilize validation data
    prediction = model.predict(X_val, batch_size=1)

    roundedPred = np.around(prediction,0)

    if multiple_targets == False and roundedPred.ndim == 1:
        i = 0
        for vals in roundedPred:
            if int(vals) == -0:
                vals = abs(vals)
                roundedPred[i] = vals

            i = i + 1
    else:
        preShape = roundedPred.shape

        # if array has multiple dimensions, flatten the array
        roundedPred = roundedPred.flatten()

        i = 0
        for vals in roundedPred:
            if int(vals) == -0:
                vals = abs(vals)
                roundedPred[i] = vals

            i = i + 1

        if len(preShape) == 3:
            if preShape[2] == 1:
                # reshape array to previous shape without the additional dimension
                roundedPred = np.reshape(roundedPred, preShape[:2])
            else:
                roundedPred = np.reshape(roundedPred, preShape)
        else:
            roundedPred = np.reshape(roundedPred, preShape)

    print("Validation Metrics")
    print("- - - - - - - - - - - - - Unrounded Prediction - - - - - - - - - - - - -")
    print(prediction)
    print("- - - - - - - - - - - - - Rounded Prediction - - - - - - - - - - - - -")
    print(roundedPred)
    print("- - - - - - - - - - - - - y val - - - - - - - - - - - - -")
    print(y_val)

    if str(type(prediction)) == "<class 'list'>":
        prediction = np.array([prediction])

    percentAcc = percentageAccuracy(roundedPred, y_val)

    print("- - - - - - - - - - - - - Percentage Accuracy - - - - - - - - - - - - -")
    print(percentAcc)

    resultList.append(str(prediction))
    resultList.append(str(roundedPred))
    resultList.append(str(y_val))
    resultList.append(str(percentAcc))

    # utilize test data
    prediction = model.predict(X_test,batch_size=1)

    roundedPred = np.around(prediction,0)

    if multiple_targets == False and roundedPred.ndim == 1: 
        i = 0
        for vals in roundedPred:
            if int(vals) == -0:
                vals = abs(vals)
                roundedPred[i] = vals

            i = i + 1
    else: 
        preShape = roundedPred.shape

        # if array has multiple dimensions, flatten the array 
        roundedPred = roundedPred.flatten()

        i = 0 
        for vals in roundedPred: 
            if int(vals) == -0: 
                vals = abs(vals)
                roundedPred[i] = vals 
            
            i = i + 1 

        if len(preShape) == 3: 
            if preShape[2] == 1: 
                # reshape array to previous shape without the additional dimension
                roundedPred = np.reshape(roundedPred,preShape[:2])
            else: 
                roundedPred = np.reshape(roundedPred,preShape)
        else: 
            roundedPred = np.reshape(roundedPred,preShape)

    print("Test Metrics")
    print("- - - - - - - - - - - - - Unrounded Prediction - - - - - - - - - - - - -")
    print(prediction)
    print("- - - - - - - - - - - - - Rounded Prediction - - - - - - - - - - - - -")
    print(roundedPred)
    print("- - - - - - - - - - - - - y test - - - - - - - - - - - - -")
    print(y_test)

    if str(type(prediction)) == "<class 'list'>":
        prediction = np.array([prediction])

    percentAcc = percentageAccuracy(roundedPred,y_test)
    
    print("- - - - - - - - - - - - - Percentage Accuracy - - - - - - - - - - - - -")
    print(percentAcc)

    resultList.append(str(prediction))
    resultList.append(str(roundedPred))
    resultList.append(str(y_test))
    resultList.append(str(percentAcc))

    if multiple_targets == True and str(type(isBinary)) == "<class 'list'>": 
        
        # initialize var as error message
        decodedPrediction = "One or all of the target variables are non-binary and/or numeric"

        i = 0
        for bools in isBinary: 
            if bools == True: 
                decodedPrediction = decode(prediction[0,i],targetDict)
            i = i + 1     
    else: 
        if isBinary: 
            decodedPrediction = decode(prediction,targetDict)
        else: 
            decodedPrediction = "One or all of the target variables are non-binary and/or numeric"

    print("- - - - - - - - - - - - - Translated Prediction - - - - - - - - - - - - -")
    print(decodedPrediction)


if not run_img_model: 
    model = clinical.clinical(main_data, mainPath, target_variables, load_fit, save_fit, model_save_loc, num_epochs, "relu")
    model.NN()

elif run_img_model: 
    model = image_model(save_dir, main_data, target_variables, num_epochs, load_numpy_img, img_array_save, load_fit, save_fit, img_dimensions, img_id_name_loc, ID_dataset_col, useCNN, data_save_loc, save_figs, show_figs)
    model.NN()

def ValResultPage():
    root = tk.Tk()

    root.title("Results - Validation")
    root.iconbitmap("D:\Cancer_Project\Cancer_ML\cancer_icon.ico")

    # MAKE SCROLLBAR
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=1)

    canvas = tk.Canvas(main_frame)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    # Add a scrollbars to the canvas
    scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    scrollbar_x = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=canvas.xview)
    scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

    # Configure the canvas
    canvas.configure(xscrollcommand=scrollbar_x.set)
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    second_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=second_frame, anchor="nw")

    # define fonts
    titleFont = tkFont.Font(family="Georgia",size=20)
    titleColor = "#f29c2c"

    resultFont = tkFont.Font(family="Consolas",size=16)

    # ADD WIDGETS
    prediction = model.resultList[0]
    roundedPred = model.resultList[1]
    y_val = model.resultList[2]
    percentAcc = model.resultList[3]

    def placeResults(txt):
        result = tk.Label(second_frame,text=txt,font=resultFont,bg='black',fg='white')
        result.grid(pady=40)

    def destroy():
        root.quit()

    resultTitle = tk.Label(second_frame,text="Prediction",font=titleFont,fg=titleColor)
    resultTitle.grid()

    placeResults(prediction)

    resultTitle = tk.Label(second_frame,text="Rounded Prediction",font=titleFont,fg=titleColor)
    resultTitle.grid()

    placeResults(roundedPred)

    resultTitle = tk.Label(second_frame,text="y_val",font=titleFont,fg=titleColor)
    resultTitle.grid()

    placeResults(y_val)

    resultTitle = tk.Label(second_frame,text="Percentage Accuracy",font=titleFont,fg=titleColor)
    resultTitle.grid()

    placeResults(percentAcc)

    exitButton = tk.Button(second_frame,text="Next",font=titleFont,fg=titleColor,command=destroy)
    exitButton.grid()

    def quit_window():
        root.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW",quit_window)
    root.mainloop()

ValResultPage()

def trainResultPage():
    root = tk.Tk()

    root.title("Results - Test")
    root.iconbitmap("D:\Cancer_Project\Cancer_ML\cancer_icon.ico")

    # Make scrollbar
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=1)

    canvas = tk.Canvas(main_frame)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    # Add a scrollbars to the canvas
    scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    scrollbar_x = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=canvas.xview)
    scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

    # Configure the canvas
    canvas.configure(xscrollcommand=scrollbar_x.set)
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    second_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=second_frame, anchor="nw")

    # define fonts
    titleFont = tkFont.Font(family="Georgia", size=20)
    titleColor = "#f29c2c"

    resultFont = tkFont.Font(family="Consolas", size=16)

    # ADD WIDGETS
    prediction = model.resultList[4]
    roundedPred = model.resultList[5]
    y_test = model.resultList[6]
    percentAcc = model.resultList[7]

    def placeResults(txt):
        result = tk.Label(second_frame,text=txt,font=resultFont,bg='black',fg='white')
        result.grid(pady=40)

    def destroy():
        root.quit()

    resultTitle = tk.Label(second_frame, text="Prediction", font=titleFont, fg=titleColor)
    resultTitle.grid()

    placeResults(prediction)

    resultTitle = tk.Label(second_frame, text="Rounded Prediction", font=titleFont, fg=titleColor)
    resultTitle.grid()

    placeResults(roundedPred)

    resultTitle = tk.Label(second_frame, text="y_test", font=titleFont, fg=titleColor)
    resultTitle.grid()

    placeResults(y_test)

    resultTitle = tk.Label(second_frame, text="Percentage Accuracy", font=titleFont, fg=titleColor)
    resultTitle.grid()

    placeResults(percentAcc)

    exitButton = tk.Button(second_frame, text="Exit", font=titleFont, fg=titleColor, command=destroy)
    exitButton.grid()

    def quit_window():
        root.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", quit_window)
    root.mainloop()

trainResultPage()

# delete converted dicom images after use if boolean is true
if del_converted_imgs == True:
    folder = save_dir
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
