###################################################
#############  Imports
from argparse import ArgumentParser
import json
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pylab import *
import random
import sys
from PIL import Image 

##########################################
#####################  read 
def read(path):
    imgs = []
    knm = {}
    strp = str(path + "\\images") 
    print("Path to images:")
    print(strp)
    for file in sorted((os.listdir(strp))):
        file_path = f"{strp}\\{file}"
        img = cv2.imread(file_path)
        imna = f'{file}'
        knm[imna] = img
        imgs.append(img)

    imagelist = []
    gimagelist = {}
    cimagelist = {}
    for img in knm:
        imgname = img
        imgg = cv2.cvtColor(knm[img], cv2.COLOR_BGR2GRAY)
        gimagelist[imgname] = imgg
        cimagelist[imgname] = knm[imgname]
    gimagelist2={}
    for i in sorted(gimagelist):
        gimagelist2[i]=gimagelist[i]
    cimagelist2={}
    for i in sorted(cimagelist):
        cimagelist2[i]=cimagelist[i]
    
    return gimagelist2, cimagelist2

##################################################################
#####################  front

def front(g, frontxml):
    Haar_front = {}
    face_cascade = cv2.CascadeClassifier(frontxml)
    for imgs in g:
        ikg = g[imgs]
        face_cords = face_cascade.detectMultiScale(ikg, scaleFactor=1.1, minNeighbors=4 )
        Haar_front[imgs] = face_cords
    return Haar_front

############################################################################
#################################  Profile 

def profile(g, profilexml):
    Haar_profile = {}
    face_cascade = cv2.CascadeClassifier(profilexml)
    for imgs in g:
        ikg = g[imgs]
        face_cords = face_cascade.detectMultiScale(ikg, scaleFactor=1.1, minNeighbors=4 )
        Haar_profile[imgs] = face_cords
    return Haar_profile


###########################################################################################
###################################   Profile flip
def profileflip(g, profilexml):
    Haar_flip = {}
    face_cascade = cv2.CascadeClassifier(profilexml)
    for imgs in g:
        ikg = g[imgs]
        ikgf = cv2.flip(ikg, 1)
        face_cords = face_cascade.detectMultiScale(ikgf, scaleFactor=1.1, minNeighbors=4 )
        Haar_flip[imgs] = face_cords
    
    for imgs in g:
        if(len(Haar_flip[imgs]) > 0):
            ikg = g[imgs]
            imw = g[imgs].shape[1]
            for faces in range(len(Haar_flip[imgs])):
                x = Haar_flip[imgs][faces][0]
                w = Haar_flip[imgs][faces][2]
                temp = x+w
                actualx = imw - temp
                Haar_flip[imgs][faces][0] = actualx
    return Haar_flip

##################################################################################################
########################################################   Face recognition 
def Facerec(g, Haar_front, Haar_profile, Haar_flip):
    Allface = {}
    for img in Haar_front:
        ikg = g[img]
        imw = g[img].shape[1]+1
        imh = g[img].shape[0]+1
        dim = (imh,imw)
        flagmat = {}
        for row in range(imh):
            for col in range(imw):
                flagmat[(row,col)] = 0
        #print(col)
        Faceimg = {}
        cordlist = []
        flagvar = {}
        for k in range(100):
            flagvar[k] = True
        k = 0
        #########################################################################
        ###########   Front    ##########################################
        if(len(Haar_front[img]) > 0):
            ikg = g[img]
            imw = g[img].shape[1]
            for faces in range(len(Haar_front[img])):
                matcharea = 0
                x = Haar_front[img][faces][0]
                y = Haar_front[img][faces][1]
                w = Haar_front[img][faces][2]
                h = Haar_front[img][faces][3]
                area = w*h
                for key in flagmat:
                    r1 = key[0]
                    c1 = key[1]
                    if(r1 >= y and r1 <= y+h and c1 >= x and c1 <= x+w):
                        if(flagmat[key] == 1):
                            matcharea += 1
                if(matcharea >= 0.5*area):
                    flagvar[faces] = False
                    
                if(flagvar[faces] == True):
                    cordlist.append(Haar_front[img][faces][0])
                    cordlist.append(Haar_front[img][faces][1])
                    cordlist.append(Haar_front[img][faces][2])
                    cordlist.append(Haar_front[img][faces][3])
 
                for key in flagmat:
                    r1 = key[0]
                    c1 = key[1]
                    if(r1 >= y and r1 <= y+h and c1 >= x and c1 <= x+w):
                        flagmat[(r1,c1)] = 1
                    
        ############################################################################
        #########     Profile #######################################
        if(len(Haar_profile[img]) > 0):
            ikg = g[img]
            imw = g[img].shape[1]
            for faces in range(len(Haar_profile[img])):
                matcharea = 0
                x = Haar_profile[img][faces][0]
                y = Haar_profile[img][faces][1]
                w = Haar_profile[img][faces][2]
                h = Haar_profile[img][faces][3]
                area = w*h
                for key in flagmat:
                    r1 = key[0]
                    c1 = key[1]
                    if(r1 >= y and r1 <= y+h and c1 >= x and c1 <= x+w):
                        if(flagmat[key] == 1):
                            matcharea += 1
                if(matcharea >= 0.45*area):
                    flagvar[faces] = False
                       
                if(flagvar[faces] == True):
                    cordlist.append(Haar_profile[img][faces][0])
                    cordlist.append(Haar_profile[img][faces][1])
                    cordlist.append(Haar_profile[img][faces][2])
                    cordlist.append(Haar_profile[img][faces][3])

                for key in flagmat:
                    r1 = key[0]
                    c1 = key[1]
                    if(r1 >= y and r1 <= y+h and c1 >= x and c1 <= x+w):
                        flagmat[(r1,c1)] = 1
    
        ############################################################################
        #############  profile flip #############################
        flagvar = {}
        for k in range(100):
            flagvar[k] = True
        k = 0
        if(len(Haar_flip[img]) > 0):
            ikg = g[img]
            imw = g[img].shape[1]
            for faces in range(len(Haar_flip[img])):
                matcharea = 0
                x = Haar_flip[img][faces][0]
                y = Haar_flip[img][faces][1]
                w = Haar_flip[img][faces][2]
                h = Haar_flip[img][faces][3]
                area = w*h
                for key in flagmat:
                    r1 = key[0]
                    c1 = key[1]
                    if(r1 >= y and r1 <= y+h and c1 >= x and c1 <= x+w):
                        if(flagmat[key] == 1):
                            matcharea += 1
                if(matcharea >= 0.45*area):
                    flagvar[faces] = False
                       
                if(flagvar[faces] == True):
                    cordlist.append(Haar_flip[img][faces][0])
                    cordlist.append(Haar_flip[img][faces][1])
                    cordlist.append(Haar_flip[img][faces][2])
                    cordlist.append(Haar_flip[img][faces][3])
   
                for key in flagmat:
                    r1 = key[0]
                    c1 = key[1]
                    if(r1 >= y and r1 <= y+h and c1 >= x and c1 <= x+w):
                        flagmat[(r1,c1)] = 1

                                  
        Allface[img] = cordlist
    return Allface

#######################################################################################
#####################################  Create result

def Finalres(Allface):
    result = []
    for imgs in Allface:
        ikg = g[imgs]
        lis = Allface[imgs]
        numberoffaces = len(lis)/4
        for nface in range(int(numberoffaces)):
            dictim = {}
            dictim["iname"] = imgs
            bb = []
            x = lis[nface*4+0]
            y = lis[nface*4+1]
            w = lis[nface*4+2]
            h = lis[nface*4+3]
            bb.append(int(x))
            bb.append(int(y))
            bb.append(int(w))
            bb.append(int(h))
            dictim["bbox"] = bb
            result.append(dictim)
            cv2.rectangle(ikg, (x,y), (x+w, y+h), (255, 0,0), thickness=2)
    return result




#############################################################################
####################################################################  main
path = str(sys.argv[1])
p = path
g, c = read(p)
frontxml = "Model_Files\\haarcascade_frontalface_alt.xml"
profilexml = "Model_Files\\haarcascade_profileface.xml"
Haar_front = front(g, frontxml)
Haar_profile = profile(g, profilexml)
Haar_flip = profileflip(g, profilexml)
Allface = Facerec(g, Haar_front, Haar_profile, Haar_flip)
result = Finalres(Allface)
results = []
for elems in result:
    results.append(elems)
output_json = "results.json"
#dump json_list to result.json
with open(output_json, 'w') as f:
    json.dump(results, f)

