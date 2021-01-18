from cv2 import cvtColor, resize, imread, rectangle
from cv2 import COLOR_BGR2GRAY, COLOR_GRAY2BGR, error
from numpy import sum as npsum
from torch import from_numpy
from numpy import exp as npexp
from numpy import array, nan, copy, rollaxis
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, imshow, imsave, show
from scipy.signal import argrelmin,argrelmax
from numpy import transpose
from torch import load, tensor, argmax
from machine import mach1, mach2
from machine import normalize

def smooth(scatter, shift=4, kerneltype="uniform", deviation=4):
    smoothed = []
    if kerneltype == "uniform":
        kernel = array([1] * shift)
    elif kerneltype == "exp":
        kernel = npexp(-array(range(shift)) / deviation)
    elif kerneltype == "gaussian":
        kernel = npexp(-(array(range(shift)) - shift / 2) ** 2 / deviation ** 2 / 2)
    else:
        print("no such kernetype as", kerneltype)
        return nan
    for j in range(-shift + 1, len(scatter)):
        loc_func = scatter[max(j, 0):min(j + shift, len(scatter))]
        if j < 0:
            smoothed.append(npsum(loc_func * kernel[-len(loc_func):] / npsum(kernel[-len(loc_func):])))
        elif j + shift > len(scatter):
            smoothed.append(npsum(loc_func * kernel[:len(loc_func)]) / npsum(kernel[:len(loc_func)]))
        else:
            smoothed.append(npsum(loc_func * kernel) / npsum(kernel))
    smoothed = smoothed[0:len(scatter)]
    return array(smoothed)

def segment_lines(im, axis=0, shift=50, kerneltype="gaussian", deviation=10):
    gray = cvtColor(im, COLOR_BGR2GRAY)
    scatter = npsum(gray, axis=axis)
    scatter = smooth(scatter, shift=shift, kerneltype=kerneltype, deviation=deviation)
    #scatter = scatter# - npmean(scatter)
    mins = argrelmin(scatter)
    answer = []
    for i in range(len(mins[0])-1):
        y1 = max(mins[0][i]-shift//8,0)
        y2 = min(mins[0][i+1]+shift//8,im.shape[1-axis])
        answer.append((y1,y2))
        #plt.imshow(im[y1:y2, :, :])
        #plt.axis("off")
        #plt.imsave("src/try_"+str(i)+".png",im[y1:y2, :, :])
    return answer


def segment_image(im, models, name):
    answer = []
    model1 = models[0]
    model2 = models[1]
    model3 = models[2]
    index=0
    draw = copy(im)
    lines = segment_lines(im,axis=1,shift=40,kerneltype="uniform")#,deviation=4*10)
    for line in lines:
        #print(line)
        answer.append([])
        segments = segment_lines(im[max(line[0]-10,0):min(line[1]+10,1600),:,:],axis=0, shift=100,kerneltype="gaussian",deviation=200)
        for segment in segments:
            if line[1]-line[0]>30 and segment[1]-segment[0]>5:
                answer[-1].append((max(line[0]-10,0),max(segment[0]-10,0),min(line[1]+10,1600),min(segment[1]+10,1200)))
                #print(answer)
                draw = rectangle(draw,(answer[-1][-1][1],answer[-1][-1][0]),(answer[-1][-1][3],answer[-1][-1][2]), (255,0,0), 3)
                box = im[answer[-1][-1][0]:answer[-1][-1][2], answer[-1][-1][1]:answer[-1][-1][3], :]
                if not (box is None):
                    try:
                        resized_box = resize(box,(100,100))
                        box = from_numpy(normalize(rollaxis(resized_box.copy(),2,0))).float().view(-1,3,100,100)
                        answer[-1][-1] = [argmax(model1(box).detach()), argmax(model2(box).detach()), argmax(model3(box)).detach()]
                        print(answer[-1][-1][0], answer[-1][-1][1], answer[-1][-1][2]//42, answer[-1][-1][2]%42)
                        print(name+"_"+str(int(answer[-1][-1][0]))+"_"+str(int(answer[-1][-1][1]))+"_"+str(int(answer[-1][-1][2])//41)+"_"+str(int(answer[-1][-1][2]%41))+"_"+str(index)+".png", "OK")
                        plt.imsave(name+"_"+str(int(answer[-1][-1][0]))+"_"+str(int(answer[-1][-1][1]))+"_"+str(int(answer[-1][-1][2]//41))+"_"+str(int(answer[-1][-1][2]%41))+"_"+str(index)+".png",resized_box)
                    except error:
                        print(name+"_"+str(index)+".png", "exception")
                        plt.imsave(name+"_"+str(index)+".png",box)
                else:
                    print("-------------------------------")
                    print(box, "in", answer[-1][-1][0],answer[-1][-1][2], answer[-1][-1][1],answer[-1][-1][3])
                    print("-------------------------------")
                plt.imsave(name+"_draw.png", draw)
                index+=1
    return answer


def give_predictions(models, image):
    im = resize(image, (1200,1600))
    answer1 = []
    answer2 = []
    answer3 = []
    for line in segment_image(im, models,"samples/sample_"+str(i)):
        answer1.append([])
        answer2.append([])
        answer3.append([])
        for x in line:
            answer1[-1].append(x[0])
            answer2[-1].append(x[1])
            answer3[-1].append(x[2])
            print(answer1)
            print(answer2)
            print(answer3)
    return (answer1, answer2, answer3)

model1 = load("model1_200_epochs.m")
model2 = load("model2_200_epochs.m")
model3 = load("model3_200_epochs.m")

for i in range(15,29):
    im = imread("TEST_IMG_"+str(i)+".jpg")
    print("processing", "TEST_IMG_"+str(i)+".jpg")
    give_predictions([model1,model2,model3],im)