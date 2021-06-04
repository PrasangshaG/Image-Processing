#######################################
##########    imports 
from argparse import ArgumentParser
import json
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pylab import *
import random
from PIL import Image 
import sys
import face_recognition

##############################################
############   Read  

def read(path):
    imgs = []
    knm = {}
    print("Path to the folder:")
    print(path)
    for file in sorted(os.listdir(path)):
        file_path = f"{path}\\{file}"
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
        
    str1 = path

    for i in range(len(str1)):
        if(str1[i] == 'f' and str1[i+1] == 'a' and str1[i+2] == 'c' and str1[i+3] == 'e' and str1[i+4] == 'C' and  str1[i+5] == 'l'):
            break
    k = int(str1[i+12])
    
    return gimagelist2, cimagelist2, k

#####################################################################
####################  Return all the faces

def returnfaces(gimglist, front):
    face_cascade = cv2.CascadeClassifier(front)
    onlyface = {}
    f = {}
    for imgname in gimglist:
        dicto = {}
        img = gimglist[imgname]
        face_cords = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
        if(len(face_cords)> 1): ## More than one face
            areas = {}
            for i in range(len(face_cords)):
                l = []
                xi = face_cords[i][0]
                yi = face_cords[i][1]
                wi = face_cords[i][2]
                hi = face_cords[i][3]
                area = wi*hi
                areas[i] = area
            maxindex = max(areas, key=areas.get)
            val = []
            #print(face_cords[maxindex][0])
            #for x, y, w, h in face_cords[maxindex]:
            valx = face_cords[maxindex][0]
            val.append(valx)
            valy = face_cords[maxindex][1]
            val.append(valy)
            valw = face_cords[maxindex][2]
            val.append(valw)
            valh = face_cords[maxindex][3]
            val.append(valh)
            f[imgname] = val
                
        else:
            val = []
            for x, y, w, h in face_cords:
                valx = x
                val.append(valx)
                valy = y
                val.append(valy)
                valw = w
                val.append(valw)
                valh = h
                val.append(valh)
            f[imgname] = val
    return f    
        

##########################################################
####################  Features 

def createfeatures(c, g, front):
    colimages = {}
    for im in c:
        colimages[im] = c[im]
    
    vect = {}
    i = 0
    for img in colimages:
        imgname = img
        iml = {}
        iml[imgname] = cv2.cvtColor(colimages[img], cv2.COLOR_BGR2GRAY)
        fll = []
        f = returnfaces(iml, front)
        loc = (f[imgname][1], f[imgname][0]+f[imgname][2],  f[imgname][1]+f[imgname][3], f[imgname][0])
        fll.append(loc)
        idk = face_recognition.face_encodings(colimages[img], fll)
        vect[imgname] = idk
        i += 1

    data = {}
    for img in vect:
        tempu = []
        for k in range(128):
            tempu.append(vect[img][0][k]*1000)
        data[img] = tempu
    
    dataarangeddiff = {}
    for k in range(128):
        tempul = []
        for img in vect:
            tempul.append(vect[img][0][k]*1000)
        dataarangeddiff[k]=tempul
    
    return data, dataarangeddiff


###############################################################
################################################################
###########      K means    #############################

################################# Initialization
def initmeans(data, k, d2):
    kmnb = []
    for f in range(len(d2)):
        feat = d2[f]
        kmeans = []
        qua = 1/(k+5)
        quar = round(qua, 3) - 0.001
        for km in range(3, k+3):
            kq = km * quar
            pointm = np.quantile(feat, kq)
            kmeans.append(pointm)
        kmnb.append(kmeans)
    inimeans = []
    for j in range(k):
        mean = []
        for i in range(len(kmnb)):
            mean.append(kmnb[i][j])
        inimeans.append(mean)

    return inimeans  

########################################  

def runKmeans(k, data, data2):
    totiter=1000
    kminsA = initmeans(data, k, data2)
    
    sizeofcluster = {}
    itemincluster = {}
    for i in range(k):
        sizeofcluster[i] = 0
    for i in range(len(data)):
        itemincluster[i] = 0
        
    for ite in range(totiter):
        for img in data:
            cluster = findcluster(kminsA, data[img])
            #print(cluster)
            sizeofcluster[cluster] += 1
            cs = sizeofcluster[cluster]
            kminsA[cluster] = uplift(kminsA[cluster], data[img])
            itemincluster[img] = cluster

    return kminsA

#########################################################

def findcluster(kmins, elem):
    mintu = 99999
    idx = -1
    for i in range(len(kmins)):
        lam = kmins[i]
        totd = 0
        for j in range(len(lam)):
            totd += pow(pow(lam[j] - elem[j], 2), 0.5)
        if(totd < mintu):
            mintu = totd
            idx = i
    return idx

###################################################### update means

def uplift(m, e):
    for i in range(len(m)):
        mi = m[i]
        mi = (mi+e[i])/2
        m[i] = mi
    return m

#####################################################   

def Exec(mns, vect):
    final = {}
    result = [[] for i in range(len(mns))] 
    for img in vect:
        cid = findcluster(mns, vect[img])
        result[cid].append(vect[img])
        imgname = img
        final[imgname] = cid
    jsonlist = []
    for c in range(k):
        dictu = {}
        dictu["cluster_no"] = c
        listimg = []
        img = 0
        for img in vect:
            imgname = img
            if(final[imgname] == c):
                listimg.append(imgname)
        dictu["elements"] = listimg
        jsonlist.append(dictu)
    return result, final, jsonlist



#############################################################################
####################################################################  main
path = str(sys.argv[1])
p = path
g, c, k = read(path)
frontxml = "Model_Files\\haarcascade_frontalface_alt.xml"
data, data2 = createfeatures(c, g, frontxml)
resmean = runKmeans(k, data, data2)
clusters, final, j = Exec(resmean,data)
print(j)
with open('clusters.json', "w") as file:
    json.dump(j, file)
allmonti = []
for i in range(len(j)):
    dic = j[i]
    ellist = dic['elements']
    numim = size(ellist)
    montih = 100
    montiw = 100
    d = (montiw, montih)
    imgnm = 0
    monti = cv2.resize(c[ellist[imgnm]],d) 
    for imgnm in range(1, len(ellist)):
        im = c[ellist[imgnm]]
        w = 100
        h = 100
        dim = (w,h)
        im = cv2.resize(im, dim)
        monti = np.hstack((monti, im))
    allmonti.append(monti)