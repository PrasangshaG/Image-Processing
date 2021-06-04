from argparse import ArgumentParser
import json
import os
import glob
import cv2
import numpy as np
import random


#########################################################################
#####  Read img #######################################################

def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

##############################################################################
#######   Show img   ########################################################

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

######################################################################################################### 
#########    Identify different intensities in a gray scale image with pixel count for each intensity ##

def distinctVals(grayimage):
    distinctval = {}
    ht = grayimage.shape[0]
    wt = grayimage.shape[1]
    for i in range(ht):
        for j in range(wt):
            pixel = grayimage[i,j]
            if(pixel not in distinctval):
                distinctval[pixel] = 0
            else:
                distinctval[pixel] = distinctval[pixel] +1
    return distinctval

#####################################################################################################################
######   Gray to binary                 #############################################################################

def gr2bin(img, thresh):
    ht = img.shape[0]
    wt = img.shape[1]
    bina = np.zeros((ht,wt), np.uint8)
    for i in range(ht):
        for j in range(wt):
            if(img[i,j] > thresh):
                bina[i,j] = 255
            else:
                bina[i,j] = 0
    return bina

####################################################################################################################
######    Two pass algorithm for connected component algorithm        ######################################

def ConnectedComp(img):
    ht = img.shape[0]
    wt = img.shape[1]
    ### labels 
    labels = {}
    labels[(-10,-10)] = 0
    
    ### union find parent dict ###
    parent = {}
    
    ### first pass ###
    for i in range(ht):
        for j in range(wt):  
            if(img[i,j] == 0): ### foreground 
                p = (i,j)
                leftp = (i, j-1)
                abovep = (i-1,j)
                if(leftp not in labels and abovep not in labels):
                    ##create new label 
                    maxlab = labels[max(labels, key=labels.get)] ###
                    labels[p] = maxlab+1
                    parent[labels[p]] = -1
                elif(leftp not in labels):
                    labels[p] = labels[abovep]
                    if(labels[abovep] < parent[labels[p]] or labels[p] not in parent):
                        parent[labels[p]] = labels[abovep]
                elif(abovep not in labels):
                    labels[p] = labels[leftp]
                    if(labels[leftp] < parent[labels[p]] or labels[p] not in parent):
                        parent[labels[p]] = labels[leftp]
                else:
                    if(labels[leftp] != labels[abovep]):
                        if(labels[leftp] < labels[abovep]):
                            labels[p] = labels[leftp]
                            parent[labels[abovep]] = labels[leftp]
                        else:
                            labels[p] = labels[abovep]
                            parent[labels[leftp]] = labels[abovep]
                                
                    else:
                        labels[p] = labels[leftp]

                    
    ###### second pass
    for i in range(ht):
        for j in range(wt):
            if(img[i,j] == 0): ###foreground
                p = (i,j)
                l = labels[p]
                paren = findparent(parent, l)
                if(paren != l):
                    labels[p] = paren
                
    
    uniquelabels = set(val for val in labels.values())
    labelpixelcount = {} 
    for uniquelabel in uniquelabels:
        labelpixelcount[uniquelabel] = 0
        for i in range(ht):
            for j in range(wt):
                if(img[i,j] == 0):
                    pix = (i,j)
                    if(labels[pix] == uniquelabel):
                        labelpixelcount[uniquelabel] += 1 
                    

    return labels, uniquelabels, parent, labelpixelcount 


###################################################################################################################
#############            Union find: findparent()            ###########################################

def findparent(parent, i):
    if(parent[i] == i or parent[i] == -1):
        return i
    else:
        return findparent(parent, parent[i])

###########################################################################################################
########    Segregate the identified connected components and returns a list of images of components   #####

def segregateComp(im, labels, uniquelabels, parent):
    comp = []
    ht = im.shape[0]
    wt = im.shape[1]
    for uniquelabel in uniquelabels:
        clust = np.zeros(im.shape)
        for i in range(ht):
            for j in range(wt):
                pi = (i,j)
                if(im[i,j] == 0): ##foreground
                    if(labels[pi] == uniquelabel):
                        clust[i,j] = 255
        comp.append(clust)
    return comp

###############################################################################################################
###########   Identifying the angle and magnitude of the gradients of an image  #######################

def angmag(gimg):
    #xkernel = np.array([[-1, 0, 1], 
    #               [-2, 0, 2], 
    #               [-1, 0, 1] ])
    #xderi = cv2.filter2D(gimg,-1,xkernel)
    #ykernel = np.array([[1, 2, 1], 
    #               [0, 0, 0], 
    #               [-1, -2, -1] ])
    #yderi = cv2.filter2D(gimg,-1,ykernel)
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    xderi = cv2.Sobel(gimg, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    yderi = cv2.Sobel(gimg, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    ####  calculate angle and magnitude of gradients ###
    ang = np.zeros(gimg.shape)
    mag = np.zeros(gimg.shape)
    for i in range(gimg.shape[0]):
        for j in range(gimg.shape[1]):
            if(xderi[i,j] != 0):
                #ang[i,j] = math.degrees(math.atan(yderi[i,j]/xderi[i,j]))
                ang[i,j] = math.atan(yderi[i,j]/xderi[i,j])
                mag[i,j] = pow(pow(xderi[i,j],2)+pow(yderi[i,j],2), 0.5)
    return ang, mag



#####################################################################################################################
########    Histogram of oriented gradients            ###################################################
def gradientHist(img):
    ang, mag = angmag(img)
    Hist = {}
    bins = []
    minang = -2
    maxang = 2
    print(minang)
    bins.append(minang)
    for i in range(1, 10):
        bini = minang + ((maxang-minang)/10)*i
        bins.append(bini)
    bins.append(maxang)
    #return bins
    totmag = mag.sum()
    for k in range(10):
        magz = 0
        zbin = (bins[k], bins[k+1])
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if(ang[i][j] >= bins[k] and ang[i][j] < bins[k+1]):
                    magz += mag[i][j]
        magz = magz/totmag
        Hist[zbin] = magz
    return Hist


################################################################################################################
##########   vertical direction transitions from background to foreground     ##############################

def verticalcross(img):
    wt = img.shape[1]
    halfwd = int(wt/2)
    numcross1 = 0 ## backgrnd-->foreground
    numcross2 = 0 ## fore --> back
    for i in range(1, img.shape[0]-1):
        if(img[i-1,halfwd] == 255 and img[i, halfwd] == 0 and img[i+1,halfwd] == 0):
            numcross1 += 1
        if(img[i-1,halfwd] == 0 and img[i, halfwd] == 0 and img[i+1,halfwd] == 255):
            numcross2 += 1
    return numcross1, numcross2

##################################################################################################################
#########     Linesweep to crop components of test image and extract the coordinates     ###################

def cropcomponent(img):
    bottomrow = 0
    toprow = 0
    leftcol = 0
    rightcol = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i,j] == 255): ##foreground
                toprow = i
                break
        if(img[i,j] == 255):
            break
        
    for i in range(img.shape[0]-1, 0, -1):
        for j in range(img.shape[1]):
            if(img[i,j] == 255): ##foreground
                bottomrow = i
                break
        if(img[i,j] == 255):
            break
    for j in range(img.shape[1]):
        for i in range(img.shape[0]):
            if(img[i,j] == 255):
                leftcol = j
                break
        if(img[i,j] == 255):
            break
    for j in range(img.shape[1]-1, 0, -1):
        for i in range(img.shape[0]):
            if(img[i,j] == 255):
                rightcol = j
                break
        if(img[i,j] == 255):
            break
    newh = (bottomrow+1)-(toprow-1)
    neww = (rightcol+1)-(leftcol-1)
    newshape = (newh, neww)
    cropimg = np.zeros(newshape)
    i = toprow-1 
    for k in range(0, cropimg.shape[0]):
        j = leftcol-1
        for l in range(0, cropimg.shape[1]):
            cropimg[k,l] = img[i,j]
            j+=1
        i+=1
    return  cropimg, (toprow,leftcol), newshape


#############################################################################################################
############       Dividing an image into number of equal sized patches    ###################################

###    dividing into patches: patch height= img_height*htfraction, patch width= img_width*wtfraction  ####
def dividepatches(img, htfraction, wtfraction):
    imght = img.shape[0]
    imgwt = img.shape[1]
    ### total number of patches = 1/(htfraction*wtfraction)
    numpatch = int(1/(htfraction*wtfraction))
    ####
    patchht = int(imght*htfraction)+1
    patchwt = int(imgwt*wtfraction)+1
    #### number of rows of patches = 1/htfraction
    numrowp = int(1/htfraction)
    #### number of columns of patches = 1/wtfraction
    numcolp = int(1/wtfraction)
    ####
    rowpatches = []
    lastrowend = 0
    rowpatchshp = (patchht, imgwt)
    for k in range(numrowp):
        patchimg = np.zeros(rowpatchshp)
        for i in range(patchht):
            patchimg[i,:] = img[i+lastrowend,:]
        lastrowend = i-1
        rowpatches.append(patchimg)
    #return rowpatches
    patches = []
    patchshp = (patchht, patchwt)
    for row in rowpatches:
        lastcolend = 0
        for l in range(numcolp):
            smallimg = np.zeros(patchshp)
            for j in range(patchwt):
                smallimg[:,j] = row[:,j+lastcolend]
            lastcolend = j-1
            patches.append(smallimg)
    return patches
        

#####################################################################################################
#######   Utility function to read and store characters to be matched   ############
def ReadandStore():
    characters = {}
    img2 = read_image("C:\\Users\\Prasangsha\\Documents\\Image Processing\\Two.jpg", show=False)
    bin2 = gr2bin(img2, 200)
    characters['2'] = bin2
    
    imga = read_image("C:\\Users\\Prasangsha\\Documents\\Image Processing\\a.jpg", show=False)
    bina = gr2bin(imga, 200)
    characters['a'] = bina
    
    imgc = read_image("C:\\Users\\Prasangsha\\Documents\\Image Processing\\c.jpg", show=False)
    binc = gr2bin(imgc, 200)
    characters['c'] = binc
    
    imgd = read_image("C:\\Users\\Prasangsha\\Documents\\Image Processing\\dot.jpg", show=False)
    bind = gr2bin(imgd, 200)
    characters['dot'] = bind
    
    imge = read_image("C:\\Users\\Prasangsha\\Documents\\Image Processing\\e.jpg", show=False)
    bine = gr2bin(imge, 200)
    characters['e'] = bine
    
    return characters


##########################################################################################################
#############       Compute Sum of Squared Distance of two vectors       ########################

def computeSSD(vec1, vec2):
    SSD = 0
    pairedelem = []
    for i in range(len(vec1)):
        elem1 = vec1[i]
        elem2 = vec2[i]
        tup = (elem1, elem2)
        pairedelem.append(tup)
    for elem in pairedelem:
        t1 = elem[0]
        t2 = elem[1]
        D = pow(t1-t2,2)
        SSD+=D
    return SSD
    
###########################################################################################################
#######        Utility function to extract features               #########################

def ExtractFeatures(img):
    patches = dividepatches(img, 0.5, 0.5)
    Histpatch = []
    vertcross = []
    for patch in patches:
        hist = gradientHist(img)
        Histpatch.append(hist)
    vi1, vi2 = verticalcross(img)
    vertcross.append((vi1,vi2))
    return (Histpatch, vertcross)

######################################################################################################
#########        ENROLMENT           ####################################################

def enrollment():
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 1 : Your Enrollment code should go here.
    characters = ReadandStore()
    ####  for each character we store the histogram of the gradient of the patches  ###
    charfeatures1 = {}
    charfeatures2 = {}
    file1 = open("C:\\Users\\Prasangsha\\Documents\\Image Processing\\myfile.txt","w")#write mode
    for character in characters:
        image = characters[character]
        (Hp,vertcross) = ExtractFeatures(image)
        charfeatures1[character] = Hp
        charfeatures2[character] = vertcross
        file1.write(str(character))
        file1.write(str(charfeatures1[character]))
        file1.write(str(charfeatures2[character]))
        file1.write('\n')
        
    #raise NotImplementedError
    return charfeatures1, charfeatures2
    

#########################################################################################################
###########       DETECTION                ######################################################

def detection():
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 2 : Your Detection code should go here.
    #raise NotImplementedError
    grayimage = read_image("C:\\Users\\Prasangsha\\Documents\\Image Processing\\test_img.jpg", show=False)
    d = distinctVals(grayimage)
    img = gr2bin(grayimage, 200)
    labels, uniquelabels, parent, labelpixelcount = ConnectedComp(img)
    components = segregateComp(img, labels, uniquelabels, parent)
    
    componentloc = {}
    componentimg = {}
    index = 0
    for component in components:
        compimg, startloc, shape = cropcomponent(component)
        outputloc = (startloc, shape)
        componentloc[index] = outputloc
        componentimg[index] = compimg
        index += 1
    
    return componentloc, componentimg
    #return components
        

##########################################################################################################################
#########     RECOGNITION       ######################################################################################

def recognition(componentloc, componentimg, charfeatures1, charfeatures2):
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.

    #raise NotImplementedError
    compfeatur = {}
    for index in componentimg:
        image = componentimg[index]
        componentfeature = ExtractFeatures(image) ## feature is a tuple: histogram and vertical cross
        compfeatur[index] = componentfeature
    
    ### character features are to be matched with component feature or test fearture ####
    SSDscores1 = {}
    SSDscores2 = {}
    for character in charfeatures1:
        charfeat = charfeatures1[character]
        for index in compfeatur:
            ssdof = (character,index)
            testtup = compfeatur[index] ## it is a tuple first element is histogram
            testfeat = testtup[0]
            ### these features are list of dictionaries. number of dictionaries = patches. dictionaries have angle wise histogram
            TotSSD = 0
            for i in range(len(testfeat)):
                patch_char = charfeat[i]
                patch_test = testfeat[i]
                vec1 = []
                vec2 = []
                for angles in patch_char:
                    vec1.append(patch_char[angles])
                    vec2.append(patch_test[angles])
                SSD = computeSSD(vec1, vec2)
                TotSSD += SSD
            SSDscores1[ssdof] = TotSSD
    
    for character in charfeatures2:
        charfeat = charfeatures2[character]
        for index in compfeatur:
            ssdof = (character,index)
            testtup = compfeatur[index] ## it is a tuple first element is histogram
            testfeat = testtup[1]
            ### these features are list of tuples. number of tuples = patches+1. tuples have vertical cross 255->0 and 0->255
            TotSSD = 0
            for i in range(len(testfeat)):
                patch_char = charfeat[i]
                patch_test = testfeat[i]
                vec1 = []
                vec2 = []
                vec1.append(patch_char[0])
                vec2.append(patch_test[0])
                for j in range(len(vec1)):
                    TotSSD += pow(vec1[j] - vec2[j],2)
                    
                #D1 = computeSSD(vec1, vec2)
                vec3 = []
                vec4 = []
                vec3.append(patch_char[1])
                vec4.append(patch_test[1])
                for j in range(len(vec3)):
                    TotSSD += pow(vec3[j] - vec4[j],2)
                #SSD2 = computeSSD(vec3, vec4)
                #SSD = SSD1+SSD2
                #TotSSD += SSD
            SSDscores2[ssdof] = TotSSD
    #return SSDscores1, SSDscores2
    
    results = []
    for ket in SSDscores1:
        if(SSDscores1[ket] < 0.046):
            if(SSDscores2[ket]==2 or SSDscores2[ket] == 1):
                charname = ket[0]
                charindex = ket[1]
                loc = componentloc[charindex]
                (x,y) = loc[0]
                (h,w) = loc[1]
                theloc = []
                theloc.append(y)
                theloc.append(x)
                theloc.append(w)
                theloc.append(h)
                dicti = {}
                dicti['bbox'] = theloc
                dicti['name'] = charname
                results.append(dicti)
        else:
            charindex = ket[1]
            loc = componentloc[charindex]
            (x,y) = loc[0]
            (h,w) = loc[1]
            theloc = []
            theloc.append(y)
            theloc.append(x)
            theloc.append(w)
            theloc.append(h)
            dicti = {}
            dicti['bbox'] = theloc
            dicti['name'] = 'UNKNOWN'
            results.append(dicti)
    return results
          
                    
##################################################################################################################
######################   OCR   #####################################################################

def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments

    cf1, cf2 = enrollment()

    cloc, cimg = detection()
    
    Results = recognition(cloc, cimg, cf1, cf2)
    
    return Results

    #raise NotImplementedError

#########################################################################################################################
#######################

def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = list(coordinates)
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)

##########################################################################################################################
############################

def main():
    #parser = argparse.ArgumentParser()
    #args = parser.parse_args()
    
    characters = []

    #all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    #for each_character in all_character_imgs :
    #    character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
    #    characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image("C:\\Users\\Prasangsha\\Documents\\Image Processing\\test_img.jpg", show=False)

    results = ocr(test_img, characters)
    print(results)

    save_results(results, "C:\\Users\\Prasangsha\\Documents\\Image Processing\\")
    

#######################
if __name__ == "__main__":
    main()




