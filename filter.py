# -*- coding: utf-8 -*-
from __future__ import division
import re
import sys
import cPickle as pickle
import os
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import random

# 0 -> ham
# 1 -> spam
def main():
    n = len(sys.argv)

    if n == 2 and sys.argv[1] == "-t":

        if os.path.isfile("knowledgebase.p"):
            os.remove("knowledgebase.p")
        knowledgebase = buildTrainingSet(400,100)
        buildLinearClassifier(400,100,knowledgebase)
        with open("knowledgebase.p","wb") as handle:
            pickle.dump(knowledgebase, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif n ==2  and sys.argv[1] == "-tAll":
        with open("knowledgebase.p","rb") as handle:
            knwb = pickle.load(handle)
        m,c = readLinearFile()
        visualize(m,c,knwb)
    elif n ==2  and sys.argv[1] == "-simpleAcc":
        testMultipleSimpleBayes()
    elif n ==2  and sys.argv[1] == "-ca":
        testMultipleComplexBayes()
    elif n == 2:
        with open("knowledgebase.p","rb") as handle:
            knwb = pickle.load(handle)
        m,c = readLinearFile()
        if os.path.isfile(sys.argv[1]):
            print classify(sys.argv[1],knwb,m,c) +"\n"
        else:
            print sys.argv[1] +" was not found!\n"

    else:
        print "Usage: -t    Build Training Set\n"


def testsymtoletSingle(filename):
    if os.path.isfile(filename):
        rFile = open(filename,'r' )
        fileText = rFile.read()
        if "<html>" in fileText or "<HTML>" in fileText:
            return -0.5
        nospaces = re.sub(r'\s+',"",fileText)
        nospaces = nospaces.replace(",","")
        nospaces = nospaces.replace(".","")
        allcount = len(nospaces)
        symbolCount = len(re.sub(r'[A-za-z]',"",nospaces))
        characterCount = allcount-symbolCount
        prop = symbolCount/characterCount
        return symbolCount/characterCount
    return -1

def laplaceCorrection(knwb):
    newD = dict()
    for key in knwb:
        if key[1] == 's':
            if not((key[0],'h') in knwb):
                newD[(key[0],'h')] = 1
        else:
            if not((key[0],'s') in knwb):
                newD[(key[0],'s')] = 1
        newD[key] = knwb[key] +1
    return newD

def buildTrainingSet(nham,nspam):
    knowledgebase = dict()
    wordsHam =0
    wordsSpam =0
    hamsProb = []
    hamsbang = []
    spamsProb = []
    spamsbang = []

    for i in range(nham):
        filename = "public/ham"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            rFile = open(filename,'r' )
            for line in rFile:
                nline = re.sub(r'[^A-Za-z ]',"",line)
                words = nline.split(' ')
                for w in words:
                    if w != "":
                        if (w,'h') in knowledgebase:
                            knowledgebase[(w,'h')] +=1
                        else:
                            knowledgebase[(w,'h')] = 1
                        wordsHam +=1

    for i in range(nspam):
        filename = "public/spam"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            rFile = open(filename,'r' )
            for line in rFile:
                nline = re.sub(r'[^A-Za-z ]',"",line)
                words = nline.split(' ')
                for w in words:
                    if w != "":
                        if (w,'s') in knowledgebase:
                            knowledgebase[(w,'s')] +=1
                        else:
                            knowledgebase[(w,'s')] = 1#
                        wordsSpam +=1


    knowledgebase = laplaceCorrection(knowledgebase)
    totWords = len(knowledgebase)/2

    knwb = dict()

    for key in knowledgebase:
        # do not count words that occur less than  5 times in total
        if knowledgebase[(key[0],'s')] +knowledgebase[(key[0],'h')] < 7:
            continue

        pws = knowledgebase[(key[0],'s')] / (wordsSpam + totWords)
        pwh = knowledgebase[(key[0],'h')] /(wordsHam +totWords)

        # ignore words with similar probabilities for spam and ham
        if (abs(pws-pwh)> max(pws,pwh)*0.40):
            knwb[key[0],'s'] = pws
            knwb[key[0],'h'] = pwh
    return knwb

def buildLinearClassifier(trainsH,trainS,knwb):
    hamsProb = []
    hamsbang = []
    spamsProb = []
    spamsbang = []

    for i in trainsH:
        filename = "public/ham"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            prob, bang = probability_and_proportion(filename,knwb)
            if (bang >=0):
                hamsProb.append(prob)
                hamsbang.append(bang)


    for i in trainS:
        filename = "public/spam"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            prob, bang = probability_and_proportion(filename,knwb)
            if (bang >=0):
                spamsProb.append(prob)
                spamsbang.append(bang)

    center1 = [np.mean(hamsProb),np.mean(hamsbang)]
    center2 =[np.mean(spamsProb),np.mean(spamsbang)]
    m,c = findLinearBoundry(center1,center2)
    return m,c

def probability_and_proportion(file, knwb):
    # getting words in email
    spamScore =0
    hamScore =0
    rFile = open(file,'r' )
    val = testsymtoletSingle(file)
    for line in rFile:
        nline = re.sub(r'[^A-Za-z ]',"",line)
        words = nline.split(' ')
        for w in words:
            if w != "":
                if (w,'s') in knwb:
                    spamScore += math.log(knwb[(w,'s')])
                    hamScore += math.log(knwb[(w,'h')])
    rFile.close()
    spamScore += math.log(1/5)
    hamScore += math.log(4/5)

    return hamScore/spamScore , val

def findLinearBoundry(center1, center2):
    difx = abs(center1[0]-center2[0])
    dify = abs(center1[1]-center2[1])
    mOrg = dify/difx
    m = -1/mOrg
    x = min(center1[0],center2[0])+difx/2
    y = min(center1[1],center2[1])+dify/2

    c = y-m*x
    return m,c

def writeLinearFile(m,c):
    wFile = open('linear.txt','w')
    wFile.write(str(m))
    wFile.write("\n")
    wFile.write(str(c))
    wFile.close()

def readLinearFile():
    rFile = open('linear.txt','r')
    m = float(rFile.readline())
    c = float(rFile.readline())
    return m,c

def linearPredict(p,m,c):
    liney = m*p[0]+c
    if (p[1] >liney):
        return "spam"
    return "ham"

def classify(filename,knwb,m,c):
    ps, ratio = probability_and_proportion(filename,knwb)
    if ratio >= 0:
        return linearPredict([ps,ratio],m,c)
    elif ps >=1:
        return "spam"
    return "ham"

def visualize(m,c,knwb):

    fig = plt.figure()
    for i in range(400):
        filename = "public/ham"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            prob, bang = probability_and_proportion(filename,knwb)
            plt.scatter(prob,bang, c='b')


    for i in range(100):
        filename = "public/spam"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            prob, bang = probability_and_proportion(filename,knwb)
            plt.scatter(prob,bang, c='r')




    for i in range(100):
        filename = "automarker/emails/spam"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            prob, bang = probability_and_proportion(filename,knwb)
            l = classify(filename,knwb,m,c)
            plt.scatter(prob,bang, c= 'r' if l =="spam" else 'b', marker= 'x',s=50)

    for i in range(400):
        filename = "automarker/emails/ham"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            prob, bang = probability_and_proportion(filename,knwb)
            l = classify(filename,knwb,m,c)
            plt.scatter(prob,bang, c= 'r' if l =="spam" else 'b', marker= 'D',s=50)

    xx = np.linspace(0,2,10)
    yy = xx*m+c
    plt.plot(xx,yy,"k-",linewidth=3)
    plt.xlim(0,2)
    plt.ylim(-1,2)
    plt.grid(True)
    plt.show()

def getAccuracyNaivebayes(trainsH,trainS,testH,testS):
    knowledgebase = dict()
    wordsHam =0
    wordsSpam =0
    hamsProb = []
    hamsbang = []
    spamsProb = []
    spamsbang = []


    for i in trainsH:
        filename = "public/ham"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            rFile = open(filename,'r' )
            for line in rFile:
                nline = re.sub(r'[^A-Za-z ]',"",line)
                words = nline.split(' ')
                for w in words:
                    if w != "":
                        if (w,'h') in knowledgebase:
                            knowledgebase[(w,'h')] +=1
                        else:
                            knowledgebase[(w,'h')] = 1
                        wordsHam +=1

    for i in trainS:
        filename = "public/spam"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            rFile = open(filename,'r' )
            for line in rFile:
                nline = re.sub(r'[^A-Za-z ]',"",line)
                words = nline.split(' ')
                for w in words:
                    if w != "":
                        if (w,'s') in knowledgebase:
                            knowledgebase[(w,'s')] +=1
                        else:
                            knowledgebase[(w,'s')] = 1#
                        wordsSpam +=1


    knowledgebase = laplaceCorrection(knowledgebase)
    totWords = len(knowledgebase)/2

    knwb = dict()

    for key in knowledgebase:
        # do not count words that occur less than  5 times in total
        pws = knowledgebase[(key[0],'s')] / (wordsSpam + totWords)
        pwh = knowledgebase[(key[0],'h')] /(wordsHam +totWords)

        knwb[key[0],'s'] = pws
        knwb[key[0],'h'] = pwh
    hamsPredict = []
    spamsPredict = []
    for i in testH:
        filename = "public/ham"+("%03d"%i)+".txt"
        hamsPredict.append(simpleBayes(knwb,filename))
    for i in testS:
        filename = "public/spam"+("%03d"%i)+".txt"
        spamsPredict.append(simpleBayes(knwb,filename))
    TP = spamsPredict.count("spam")
    TN = hamsPredict.count("ham")
    FP = spamsPredict.count("ham")
    FN = hamsPredict.count("spam")
    ACC = (TP + TN)/(TP + TN + FP + FN)
    return ACC

def getAccuracyFinal(trainsH,trainS,testH,testS):
    knowledgebase = dict()
    wordsHam =0
    wordsSpam =0
    hamsProb = []
    hamsbang = []
    spamsProb = []
    spamsbang = []


    for i in trainsH:
        filename = "public/ham"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            rFile = open(filename,'r' )
            for line in rFile:
                nline = re.sub(r'[^A-Za-z ]',"",line)
                words = nline.split(' ')
                for w in words:
                    if w != "":
                        if (w,'h') in knowledgebase:
                            knowledgebase[(w,'h')] +=1
                        else:
                            knowledgebase[(w,'h')] = 1
                        wordsHam +=1

    for i in trainS:
        filename = "public/spam"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            rFile = open(filename,'r' )
            for line in rFile:
                nline = re.sub(r'[^A-Za-z ]',"",line)
                words = nline.split(' ')
                for w in words:
                    if w != "":
                        if (w,'s') in knowledgebase:
                            knowledgebase[(w,'s')] +=1
                        else:
                            knowledgebase[(w,'s')] = 1#
                        wordsSpam +=1


    knowledgebase = laplaceCorrection(knowledgebase)
    totWords = len(knowledgebase)/2

    knwb = dict()

    for key in knowledgebase:
        # do not count words that occur less than  5 times in total
        if knowledgebase[(key[0],'s')] +knowledgebase[(key[0],'h')] < 5:
            continue

        pws = knowledgebase[(key[0],'s')] / (wordsSpam + totWords)
        pwh = knowledgebase[(key[0],'h')] /(wordsHam +totWords)

        # ignore words with similar probabilities for spam and ham
        if (abs(pws-pwh)> max(pws,pwh)*0.10):
            knwb[key[0],'s'] = pws
            knwb[key[0],'h'] = pwh
    m,c = buildLinearClassifier(trainsH,trainS,knwb)
    hamsPredict = []
    spamsPredict = []
    for i in testH:
        filename = "public/ham"+("%03d"%i)+".txt"
        hamsPredict.append(classify(filename,knwb,m,c))
    for i in testS:
        filename = "public/spam"+("%03d"%i)+".txt"
        spamsPredict.append(classify(filename,knwb,m,c))
    TP = spamsPredict.count("spam")
    TN = hamsPredict.count("ham")
    FP = spamsPredict.count("ham")
    FN = hamsPredict.count("spam")
    ACC = (TP + TN)/(TP + TN + FP + FN)
    return ACC


def simpleBayes(knwb,filename):
    spamScore =0
    hamScore =0
    rFile = open(filename,'r' )
    val = testsymtoletSingle(filename)
    for line in rFile:
        nline = re.sub(r'[^A-Za-z ]',"",line)
        words = nline.split(' ')
        for w in words:
            if w != "":
                if (w,'s') in knwb:
                    spamScore += math.log(knwb[(w,'s')])
                    hamScore += math.log(knwb[(w,'h')])
    rFile.close()
    spamScore += math.log(1/5)
    hamScore += math.log(4/5)
    if (spamScore >= hamScore):
        return "spam"
    return "ham"

def testMultipleSimpleBayes():
    ACCS = []
    for i in range(20):
        train_ham, train_spam, test_hams, test_spam = divideTestTrain()
        ACCS.append(getAccuracyNaivebayes(train_ham, train_spam, test_hams, test_spam))
    print "Accuracy per iteration: {}".format(ACCS)
    print "Mean Accuracy {}".format(np.mean(ACCS))

def testMultipleComplexBayes():
    ACCS = []
    for i in range(20):
        train_ham, train_spam, test_hams, test_spam = divideTestTrain()
        ACCS.append(getAccuracyFinal(train_ham, train_spam, test_hams, test_spam))
    print "Accuracy per iteration: {}".format(ACCS)
    print "Mean Accuracy {}".format(np.mean(ACCS))


def divideTestTrain():
    hamsIs = range(400)
    spamIs = range(100)

    test_spam = []
    test_hams =[]

    train_ham =[]
    train_spam=[]

    # trainsinset size = 350 test set size = 150
    # 70S 280H                            120H 30S
    for i in range(30):
        test_spam.append(spamIs.pop(random.randint(0,len(spamIs)-1)))
        test_hams.append(hamsIs.pop(random.randint(0,len(hamsIs)-1)))
        test_hams.append(hamsIs.pop(random.randint(0,len(hamsIs)-1)))
        test_hams.append(hamsIs.pop(random.randint(0,len(hamsIs)-1)))
        test_hams.append(hamsIs.pop(random.randint(0,len(hamsIs)-1)))
    train_ham=hamsIs
    train_spam=spamIs
    return train_ham, train_spam, test_hams, test_spam

main()
