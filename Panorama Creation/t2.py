#################################################################
#####   Match SIFT Features     #################################

def matchimages(img1, img2):
    sift = cv2.SIFT_create()
    kp1,des1 = sift.detectAndCompute(img1, None)
    kp2,des2 = sift.detectAndCompute(img2, None)
    
    match = {}
    sourcept = []
    destpt = []
    sp = []
    dp = []
    for i in range(len(kp1)):
        distance = {}
        for k in range(len(kp2)):
            d = 0
            for j in range(128):
                point1 = kp1[i]
                val1 = des1[i][j]
                point2 = kp2[k]
                val2 = des2[k][j]
                d += pow(pow((val1-val2),2), 0.5)
            distance[point2] = d
        mindist = min(distance.values())
        res = [key for key in distance if distance[key] == mindist]
        min2 = 9999
        for v in distance.values():
            if(v<min2 and v>mindist):
                min2 = v
        if(mindist/min2 < 0.7):
            #print((point1.pt, res[0].pt) )
            match[point1.pt]= res[0].pt
            sourcept.append(point1.pt)
            destpt.append(res[0].pt)
            sp.append(point1)
            dp.append(res[0])
    return match, sourcept, destpt, sp, dp
            

##########################################################################
########  Create a binding box of match part   #####################

def boundingbox(match, sourcept, destpt, sp, dp, img1, img2):
    mt = 0
    flag = 1
    M = 0
    if(len(sp) > 5):
        src_pts = np.float32([sp[g].pt for g in range(len(sp))]).reshape(-1,1,2)
        dst_pts = np.float32([dp[g].pt for g in range(len(dp))]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
    
        sourcp = []
        desp = []
        ####   CONSIDER ONLY IF MATCHESMASK IS 1  ####
        j = 0
        for i in range(len(matchesMask)):
            if(matchesMask[i] == 1):
                sourcp.append(sourcept[i])
                desp.append(destpt[i])    
    
        sourcex = {}
        sourcey = {}
        for i in range(len(sourcp)):
            point = sourcp[i]
            xc = point[0]
            yc = point[1]
            sourcex[i] = xc
            sourcey[i] = yc
    
        
        ##### min value of sourcex is the left boundary; max value of source is right boundary  ####
        minx = min(sourcex.values())
        l = [index for index in sourcex if sourcex[index] == minx]
        li = l[0]
        leftpoint_s = (sourcex[li], sourcey[li])
    
        maxx = max(sourcex.values())
        r = [index for index in sourcex if sourcex[index] == maxx]
        ri = r[0]
        rightpoint_s = (sourcex[ri], sourcey[ri])
    
        ##### min value of sourcey is the bottom boundary; max value of sourcey is top boundary ####
        miny = min(sourcey.values())
        b = [index for index in sourcey if sourcey[index] == miny]
        bi = b[0]
        bottompoint_s = (sourcex[bi], sourcey[bi])
    
        maxy = max(sourcey.values())
        t = [index for index in sourcey if sourcey[index] == maxy]
        ti = t[0]
        toppoint_s = (sourcex[ti], sourcey[ti])
    
        #return leftpoint, rightpoint, bottompoint, toppoint
    
        ###  Now corresponding destination points ###
        leftpoint_d = desp[li]
        rightpoint_d = desp[ri]
        bottompoint_d = desp[bi]
        toppoint_d = desp[ti]
    
        #return leftpoint_d, rightpoint_d, bottompoint_d, toppoint_d
    
        ######  only the source image part #####
        ht = int(maxy-miny)
        wt = int(maxx-minx)
        shp = (ht,wt)
        matchpart = np.zeros(shp)
        i = int(miny) 
        for k in range(0, shp[0]):
            j = int(minx)
            for l in range(0, shp[1]):
                matchpart[k,l] = img1[i,j]
                j+=1
            i+=1
        return matchpart, M, 1
    else:
        return 0, 0, 0


######################################################################
#####    

def StitchImages(img1_g, img2_g, img1_c, img2_c, H):
    h1, w1 = img1_g.shape
    h2, w2 = img2_g.shape

    source1 = np.float32([[0,0], [0, h1],[w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    destim = np.float32([[0,0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1,1,2)
    source2 = cv2.perspectiveTransform(destim, H)
    points = np.concatenate((source1,source2), axis=0)

    [minx, miny] = np.int32(points.min(axis=0).ravel() - 0.5)
    [maxx, maxy] = np.int32(points.max(axis=0).ravel() + 0.5)
  
    transl = [-minx,-miny]
  
    H_t = np.array([[1, 0, transl[0]], [0, 1, transl[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(img2_g, H_t.dot(H), (maxx-minx, maxy-miny))
    output_img[transl[1]:h1+transl[1], transl[0]:w1+transl[0]] = img1_g
    
    output_img_c = cv2.warpPerspective(img2_c, H_t.dot(H), (maxx-minx, maxy-miny))
    output_img_c[transl[1]:h1+transl[1], transl[0]:w1+transl[0]] = img1_c

    return output_img, output_img_c 
    

###########################################################################
#############  Create Onehotarray   #####

def OnehotArray(imagelist):
    matrixsize = (len(imagelist),len(imagelist))
    matchmatrix = np.zeros(matrixsize)
    for i in range(len(imagelist)):
        isource = imagelist[i]
        sourceshp = isource.shape[0]*isource.shape[1]
        for j in range(i+1, len(imagelist)):
            idest = imagelist[j]
            match, sourcep, destpt, kp1, kp2 = matchimages(isource, idest)
            mat, n, f = boundingbox(match, sourcep, destpt, kp1, kp2, isource, idest)
            if(f == 1):
                matchshp = mat.shape[0]*mat.shape[1]
                if(matchshp/sourceshp >= 0.20):
                    matchmatrix[i,j] = 1
                    matchmatrix[j,i] = 1
                else:
                    matchmatrix[i,j] = (matchshp/sourceshp)
                    matchmatrix[j,i] = (matchshp/sourceshp)
            else:
                matchmatrix[i,j] = 0
                matchmatrix[j,i] = 0
    
    for i in range(len(imagelist)):
        for j in range(len(imagelist)):
            if(i == j):
                matchmatrix[i,j] = 1                  
    return matchmatrix


####################################################################
#######   

def Stitches(imagelist_g, imagelist_c):
    result_g = imagelist_g[0]
    result_c = imagelist_c[0]
    for j in range(1, len(imagelist_g)):
        imgdest_g = imagelist_g[j]
        imgdest_c = imagelist_c[j]
        match, sourcep, destpt, kp1, kp2 = matchimages(result_g, imgdest_g)
        src_pts = np.float32([kp1[g].pt for g in range(len(kp1))]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[g].pt for g in range(len(kp2))]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        result_g, result_c = StitchImages(imgdest_g, result_g, imgdest_c, result_c,  M)
    return result_g, result_c

##########################################################################
############

def FinalImagelist(Onehotmat, gimagelist, cimagelist):
    FinalList_gray = []
    FinalList_col = []
    sumlist = []
    for i in range(len(Onehotmat)):
        lis = Onehotmat[i]
        s = 0
        for j in range(len(lis)):
            if(i != j):
                s += Onehotmat[i,j]
        sumlist.append(s)
    for k in range(len(sumlist)):
        if(sumlist[k] != 0):
            FinalList_gray.append(gimagelist[k])
            FinalList_col.append(cimagelist[k])
    return FinalList_gray, FinalList_col


##################################################################
########

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

def stitch(imgmark, N=4, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'C:/Users/Prasangsha/Documents/Image Processing/Project2/project2/images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"
    imagelist = []
    gimagelist = []
    cimagelist = []
    for img in imgs:
        imgg = img[:,:,0]
        gimagelist.append(imgg)
        cimagelist.append(img)
    
    bm = OnehotArray(gimagelist)
    overlap_arr = bm
    Finallist_g, Finallist_c = FinalImagelist(bm, gimagelist, cimagelist)
    Final_g, Final_c = Stitches(Finallist_g, Finallist_c)

    cv2.imwrite(savepath, Final_c)
    return overlap_arr, Finallist_g


##################################################################
###########

if __name__ == "__main__":
    #task2
    overlap_arr, Fl = stitch('t2', N=4, savepath='task22.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    #bonus
    #overlap_arr2 = stitch('t3', savepath='task3.png')
    #with open('t3_overlap.txt', 'w') as outfile:
    #    json.dump(overlap_arr2.tolist(), outfile)

