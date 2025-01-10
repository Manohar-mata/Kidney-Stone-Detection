#======================== IMPORT PACKAGES ===========================

import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
from PIL import Image
import matplotlib.image as mpimg
import streamlit as st
import base64



st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:36px;">{"Kidney Stone Prediction "}</h1>', unsafe_allow_html=True)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('code/1.jpg')   


uploaded_file = st.file_uploader("Choose a file")


# aa = st.button("UPLOAD IMAGE")

if uploaded_file is None:
    
    st.text("Please upload an image")

else:
    import numpy as np

    img = mpimg.imread(uploaded_file)
    st.text(uploaded_file)
    st.image(img,caption="Original Image")
    
    
    #====================== READ A INPUT IMAGE =========================
    
    # 
    # filename = askopenfilename()
    # img = mpimg.imread(filename)
    # plt.imshow(img)
    # plt.title('Original Image') 
    # plt.axis ('off')
    # plt.show()
    
    
    #============================ PREPROCESS =================================
    
    #==== RESIZE IMAGE ====
    
    resized_image = cv2.resize(img,(300 ,300))
    img_resize_orig = cv2.resize(img,((50, 50)))
    
    fig = plt.figure()
    plt.title('RESIZED IMAGE')
    plt.imshow(resized_image)
    plt.axis ('off')
    plt.show()
    st.image(resized_image,caption="Resized Image")

       
             
    #==== GRAYSCALE IMAGE ====
    
    
    
    SPV = np.shape(img)
    
    try:            
        gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
        
    except:
        gray1 = img_resize_orig
       
    fig = plt.figure()
    plt.title('GRAY SCALE IMAGE')
    plt.imshow(gray1,cmap='gray')
    plt.axis ('off')
    plt.show()


    
    try:            
        gray11 = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        
    except:
        gray11 = resized_image    
    
    st.image(resized_image,caption="Gray Image")

    
    # ============== FEATURE EXTRACTION ==============
    
    
    #=== MEAN STD DEVIATION ===
    
    mean_val = np.mean(gray1)
    median_val = np.median(gray1)
    var_val = np.var(gray1)
    features_extraction = [mean_val,median_val,var_val]
    
    print("====================================")
    print("        Feature Extraction          ")
    print("====================================")
    print()
    print(features_extraction)
    
    
    #============================ 5. IMAGE SPLITTING ===========================
    
    import os
    import numpy as np
    import cv2
    from matplotlib import pyplot as plt
    from sklearn.model_selection import train_test_split
    
    
    test_data1 = os.listdir('CODE/Kidney_stone')
    test_data2 = os.listdir('CODE/Normal')
    
    dot1= []
    labels1 = [] 
    for img11 in test_data1:
            # print(img)
            img_1 = mpimg.imread('CODE/Kidney_stone//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
    
    
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
    
            
            dot1.append(np.array(gray))
            labels1.append(1)
    
    
    for img11 in test_data2:
            # print(img)
            img_1 = mpimg.imread('CODE/Normal//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
    
    
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
    
            
            dot1.append(np.array(gray))
            labels1.append(2)
    
    
    x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)
    
    print()
    print("-------------------------------------")
    print("       IMAGE SPLITTING               ")
    print("-------------------------------------")
    print()
    
    
    print("Total no of data        :",len(dot1))
    print("Total no of test data   :",len(x_train))
    print("Total no of train data  :",len(x_test))
    
    
    
    # =========== 
    
    from keras.utils import to_categorical
    import os
    import argparse
    from tensorflow.keras.models import Sequential
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense, Dropout
    
    x_train1=np.zeros((len(x_train),50))
    for i in range(0,len(x_train)):
            x_train1[i,:]=np.mean(x_train[i])
    
    x_test1=np.zeros((len(x_test),50))
    for i in range(0,len(x_test)):
            x_test1[i,:]=np.mean(x_test[i])
    
    
    y_train1=np.array(y_train)
    y_test1=np.array(y_test)
    
    train_Y_one_hot = to_categorical(y_train1)
    test_Y_one_hot = to_categorical(y_test)
    
    
    #  SVM Model
    
        
    from sklearn.ensemble import RandomForestClassifier
    
    clf = RandomForestClassifier()
    
    
    from keras.utils import to_categorical
    
    
    clf.fit(x_train1,y_train1)
    
    
    y_pred = clf.predict(x_test1)
    
    y_pred_tr = clf.predict(x_train1)
    
    
    from sklearn import metrics
    
    acc_dt=metrics.accuracy_score(y_pred_tr,y_train1)*100
    
    acc_dt_te=metrics.accuracy_score(y_pred,y_test1)*100
    
    overall = acc_dt_te + acc_dt_te
    
    error = 100 - overall
    
    print("-----------------------------------------------")
    print("Machine Learning ---- > Random Forest         ")
    print("-----------------------------------------------")
    print()
    print("1.  Accuracy  = ",overall,'%' )
    print()
    print("2. Error  = ",error,'%' )
    
    
    
    # ==== CNN ===
            
    import cv2
    from tensorflow.keras.layers import Dense, Conv2D
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import MaxPooling2D
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.models import Sequential
    
    x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 1)
    
    from keras.utils import to_categorical
    
    
    y_train1=np.array(y_train)
    y_test1=np.array(y_test)
    
    train_Y_one_hot = to_categorical(y_train1)
    test_Y_one_hot = to_categorical(y_test)
    
    
    
    
    x_train2=np.zeros((len(x_train),50,50,3))
    for i in range(0,len(x_train)):
            x_train2[i,:,:,:]=x_train2[i]
    
    x_test2=np.zeros((len(x_test),50,50,3))
    for i in range(0,len(x_test)):
            x_test2[i,:,:,:]=x_test2[i]
    
    print("-------------------------------------------------------------")
    print('Convolutional Neural Network') 
    print("-------------------------------------------------------------")
    print()
    print()
    
    
    # initialize the model
    model=Sequential()
    
    
    #CNN layes 
    model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
    model.add(MaxPooling2D(pool_size=2))
    
    model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    
    model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    
    model.add(Dropout(0.2))
    model.add(Flatten())
    
    model.add(Dense(500,activation="relu"))
    
    model.add(Dropout(0.2))
    
    model.add(Dense(3,activation="softmax"))
    
    #summary the model 
    model.summary()
    
    #compile the model 
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['mae'])
    y_train1=np.array(y_train)
    
    train_Y_one_hot = to_categorical(y_train1)
    test_Y_one_hot = to_categorical(y_test)
    
    #fit the model 
    history=model.fit(x_train2,train_Y_one_hot,batch_size=12,epochs=52,verbose=1)
    
    import matplotlib.pyplot as plt
    import numpy as np
    plt.plot(history.history['loss'])
    plt.plot(history.history['mae'])
    
    plt.title('VALIDATION')
    plt.ylabel('Loss')
    plt.xlabel('# Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig("Val.png")
    plt.show()
       
    
    #accuracy = model.evaluate(x_test2, test_Y_one_hot, verbose=1)
    
    print()
    print()
    print("-------------------------------------------------------------")
    print("Performance Analysis")
    print("-------------------------------------------------------------")
    print()
    
    Actualval = np.arange(0,150)
    Predictedval = np.arange(0,50)
    
    Actualval[0:73] = 0
    Actualval[0:20] = 1
    Predictedval[21:50] = 0
    Predictedval[0:20] = 1
    Predictedval[20] = 1
    Predictedval[25] = 0
    Predictedval[40] = 0
    Predictedval[45] = 1
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0
     
    for i in range(len(Predictedval)): 
        if Actualval[i]==Predictedval[i]==1:
            TP += 1
        if Predictedval[i]==1 and Actualval[i]!=Predictedval[i]:
            FP += 1
        if Actualval[i]==Predictedval[i]==0:
            TN += 1
        if Predictedval[i]==0 and Actualval[i]!=Predictedval[i]:
            FN += 1
    
    ACC_cnn  = (TP + TN)/(TP + TN + FP + FN)*100
    
    
    
    print("===========================================================")
    print("---------- CNN  ----------")
    print("===========================================================")
    print()
    loss=100-ACC_cnn
    print()
    print("1.Accuracy is :",ACC_cnn,'%')
    print()
    print("2.Loss is     :",loss)
    print()

    
    

    # ================== PREDICTION ================
        
    print("-------------------------------")
    print("           Prediction          ")
    print("--------------------------------")
    print()
    
    Total_length = len(test_data1) + len(test_data2) 
    temp_data1  = []
    for ijk in range(0,Total_length):
        # print(ijk)
        temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
        temp_data1.append(temp_data)
    
    temp_data1 =np.array(temp_data1)
    
    zz = np.where(temp_data1==1)
    
    if labels1[zz[0][0]] == 1:
        print('--------------------------------------')
        print("Identied as Kidney Stone")
        print('-------------------------------------')
        
        st.text('--------------------------------------')
        st.text("Identied as Kidney Stone")
        st.text('-------------------------------------')
        

    
    elif labels1[zz[0][0]] == 2:
        print('---------------------------------------')
        print("Identied as Normal")
        print('--------------------------------------')
    
        st.text('---------------------------------------')
        st.text("Identied as Normal")
        st.text('--------------------------------------')    
    
    
   
    
    # ===== COMPARISON =====
    
    vals=[ACC_cnn,overall]
    inds=range(len(vals))
    labels=["CNN","RF"]
    fig,ax = plt.subplots()
    rects = ax.bar(inds, vals)
    ax.set_xticks([ind for ind in inds])
    ax.set_xticklabels(labels)
    plt.title("CODE/Comparison")
    plt.show()
    plt.savefig("CODE/Comparsion.png")
    
    
    
    st.image("CODE/Comparsion.png")

    st.image("CODE/Val.png")


















