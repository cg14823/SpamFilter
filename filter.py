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
import platform

# 0 -> ham
# 1 -> spam

buzzwords = ["free","new","spam","sir","madam"]

AMOUNT = 8
SPAMICITY = 0.35

def main():
    n = len(sys.argv)

    if n == 2 and sys.argv[1] == "-t":

        if os.path.isfile("knowledgebase.p"):
            os.remove("knowledgebase.p")
        knowledgebase = buildTrainingSet(400,100)
        m,c =buildLinearClassifier(range(400),range(100),knowledgebase)
        writeLinearFile(m,c)
        with open("knowledgebase.p","wb") as handle:
            pickle.dump(knowledgebase, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif n ==2  and sys.argv[1] == "-tAll":
        with open("knowledgebase.p","rb") as handle:
            knwb = pickle.load(handle)
        m,c = readLinearFile()
        visualize(m,c,knwb)

    elif n ==2  and sys.argv[1] == "-simpleAcc":
        testMultipleSimpleBayes()
    elif n ==2  and sys.argv[1] == "-tB":
        testBoth()
    elif n ==2  and sys.argv[1] == "-ca":
        testMultipleComplexBayes()
    elif n ==2  and sys.argv[1] == "-tuneS":
        tuneSpamicity()
    elif n ==2  and sys.argv[1] == "-tuneA":
        tuneAmaount()
    elif n ==2  and sys.argv[1] == "-tune2d":
        tunner2d()
    elif n ==2  and sys.argv[1] == "-numW":
        for i in range(400):
            filename = "public/ham"+("%03d"%i)+".txt"
            val = whiteSpaceFeature(filename)
        for i in range(100):
            filename = "public/spam"+("%03d"%i)+".txt"
            val = whiteSpaceFeature(filename)
    elif n ==2  and sys.argv[1] == "-numCaps":
        for i in range(400):
            filename = "public/ham"+("%03d"%i)+".txt"
            val = capitalLetterFeature(filename)
        for i in range(100):
            filename = "public/spam"+("%03d"%i)+".txt"
            val = capitalLetterFeature(filename)
    elif n ==2  and sys.argv[1] == "-numDigs":
        for i in range(400):
            filename = "public/ham"+("%03d"%i)+".txt"
            val = digitFeature(filename)
        for i in range(100):
            filename = "public/spam"+("%03d"%i)+".txt"
            val = digitFeature(filename)
    elif n ==2  and sys.argv[1] == "-numLinks":
        for i in range(400):
            filename = "public/ham"+("%03d"%i)+".txt"
            val = linkFeature(filename)
        for i in range(100):
            filename = "public/spam"+("%03d"%i)+".txt"
            val = linkFeature(filename)
    elif n ==2  and sys.argv[1] == "-numBuzz":
        for i in range(400):
            filename = "public/ham"+("%03d"%i)+".txt"
            val = buzzwordFeature(filename)
        for i in range(100):
            filename = "public/spam"+("%03d"%i)+".txt"
            val = buzzwordFeature(filename)
    elif n ==2  and sys.argv[1] == "-PLOT":
        with open("knowledgebase.p","rb") as handle:
            knwb = pickle.load(handle)
        trainsH,trainS,testH,testS = divideTestTrain()
        m,c = buildLinearClassifierClone(trainsH,trainS,knwb)
        visualizeClone(knwb,m,c)
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


def whiteSpaceFeature(filename):
    amountWhite = 0
    numChar = 0
    if os.path.isfile(filename):
        rFile = open(filename,'r' )
        for line in rFile:
            numChar += len(line)
            nline = re.sub(r'[\S]',"",line)
            amountWhite += len(nline)

    return amountWhite/numChar

def capitalLetterFeature(filename):
    amountCaps = 0
    amountNon = 0
    numChar = 0
    if os.path.isfile(filename):
        rFile = open(filename,'r' )
        #for line in rFile:
        text = rFile.read()
        text = re.sub('\s+',"",text)
        onlyCaps = re.sub(r'[^A-Z]',"",text)
        onlyLower = re.sub(r'[^a-z]',"",text)
        amountCaps = len(onlyCaps)
        amountNon = len(onlyLower)
        numChar = amountCaps+amountNon
        #if (numChar == 0):
        #print "Caps:{0},Non:{1},Tot:{2}".format(amountCaps,amountNon,numChar)
        #print "Chars:{0}, Caps:{1}".format(numChar,amountCaps)

    return amountCaps/numChar

def digitFeature(filename):
    amountDigits = 0
    amountNon = 0
    numChar = 0
    if os.path.isfile(filename):
        rFile = open(filename,'r' )
        #for line in rFile:
        text = rFile.read()
        text = re.sub('\s+',"",text)
        onlyDigits = re.sub(r'[^0-9]',"",text)
        onlyNon = re.sub(r'[0-9]',"",text)
        amountDigits = len(onlyDigits)
        amountNon = len(onlyNon)
        numChar = amountDigits+amountNon
        #print onlyNon
        #if (numChar == 0):
        #print "Caps:{0},Non:{1},Tot:{2}".format(amountCaps,amountNon,numChar)

    return amountDigits/numChar

def linkFeature(filename):
    amountLinks = 0
    amountNon = 0
    numWords = 0
    if os.path.isfile(filename):
        rFile = open(filename,'r' )
        for line in rFile:
            words = line.split(" ")
            #text = re.sub('\s+',"",line)
            for w in words:
                if ("http" in w) or ("www." in w):
                    amountLinks += 1
                else:
                    amountNon += 1
                    #if "www" in line:
                        #print line
        numWords = amountLinks+amountNon

    #print "Links:{0},Non:{1},Tot:{2}".format(amountLinks,amountNon,numWords)
    return amountLinks/numWords

def buzzwordFeature(filename):
    amountBuzz = 0
    amountNon = 0
    numWords = 0
    if os.path.isfile(filename):
        rFile = open(filename,'r' )
        for line in rFile:
            words = line.split(" ")
            #text = re.sub('\s+',"",line)
            for w in words:
                for b in buzzwords:
                    if (b in w):
                        amountBuzz += 1
                    else:
                        amountNon += 1
                        #if "www" in line:
                            #print line
        numWords = amountBuzz+amountNon

    #print "Buzz:{0},Non:{1},Tot:{2}".format(amountBuzz,amountNon,numWords)
    return amountBuzz


def testsymtoletSingle(filename):
    # remove as much of the headers as possible
    if os.path.isfile(filename):
        rFile = open(filename,'r' )
        fileText = rFile.read()
        rFile.close()
        if "<html>" in fileText or "<HTML>" in fileText:
            return -0.5
        fileText = removeHeaders(fileText)
        nospaces = re.sub(r'\s+',"",fileText)
        nospaces = nospaces.replace(",","")
        nospaces = nospaces.replace(".","")
        allcount = len(nospaces)
        symbolCount = len(re.sub(r'[A-Za-z]',"",nospaces))
        characterCount = allcount-symbolCount
        prop = -0.5
        try:
            prop = symbolCount/characterCount
        except ZeroDivisionError:
            print "cgharactercount was 0\n"
        return prop
    return -1

def removeHeaders(str):
    i = str.find("Subject: ")
    if (i>=0):
        uptoSubject = str[i:]
        newLineI = uptoSubject.index("\n")
        subjectLine = str[i:newLineI+1]
        headers =["From:","Date:", "Sender:","Precedence:","References:"]
        textAfterSubject = uptoSubject[newLineI:]
        lines = textAfterSubject.split("\n")
        indexor = 0
        for l in lines:
            ws = l.split(" ")
            if (len(ws) > 1 ):
                w = ws[0]
                if(len(w)>1):
                    c = w[-1]
                    if not (c == ':' and ( w in headers or '-' in w)):
                        #print "here"
                        break
                else:
                    break
            else:
                break
            indexor += 1
        newLines = lines[indexor:]
        newLines = "\n".join(newLines)
        text = uptoSubject + newLines
        return text
    return str

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
        if knowledgebase[(key[0],'s')] +knowledgebase[(key[0],'h')] < AMOUNT:
            continue

        pws = knowledgebase[(key[0],'s')] / (wordsSpam + totWords)
        pwh = knowledgebase[(key[0],'h')] /(wordsHam +totWords)

        # ignore words with similar probabilities for spam and ham
        if (abs(pws-pwh)> max(pws,pwh)*SPAMICITY):
            knwb[key[0],'s'] = pws
            knwb[key[0],'h'] = pwh
    return knwb

def buildLinearClassifierClone(trainsH,trainS,knwb):
    hamsX = []
    hamsY = []
    spamsX = []
    spamsY = []

    for i in trainsH:
        filename = "public/ham"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            prob, bang = probability_and_proportion(filename,knwb)
            caps = whiteSpaceFeature(filename)
            digs = digitFeature(filename)
            links = linkFeature(filename)
            buzz = buzzwordFeature(filename)
            hamsX.append(prob)
            hamsY.append(buzz)


    for i in trainS:
        filename = "public/spam"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            prob, bang = probability_and_proportion(filename,knwb)
            caps = whiteSpaceFeature(filename)
            digs = digitFeature(filename)
            links = linkFeature(filename)
            buzz = buzzwordFeature(filename)
            spamsX.append(prob)
            spamsY.append(buzz)

    center1 = [np.mean(hamsX),np.mean(hamsY)]
    center2 =[np.mean(spamsX),np.mean(spamsY)]
    m,c = findLinearBoundry(center1,center2)
    return m,c


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

def visualizeClone(knwb,m,c):
    val1 = 0
    val2 = 0
    val3 = 0
    val4 = 0
    n1 = 0
    n2 = 0
    fig = plt.figure()
    for i in range(400):
        filename = "public/ham"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            prob, bang = probability_and_proportion(filename,knwb)
            caps = whiteSpaceFeature(filename)
            digs = digitFeature(filename)
            links = linkFeature(filename)
            buzz = buzzwordFeature(filename)
            val1 += prob
            val2 += buzz
            n1 += 1
            plt.scatter(prob,buzz, c='b')


    for i in range(100):
        filename = "public/spam"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            prob, bang = probability_and_proportion(filename,knwb)
            caps = whiteSpaceFeature(filename)
            digs = digitFeature(filename)
            links = linkFeature(filename)
            buzz = buzzwordFeature(filename)
            val3 += prob
            val4 += buzz
            n2 += 1
            plt.scatter(prob,buzz, c='r')

    for i in range(100):
        filename = "automarker/emails/spam"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            prob, bang = probability_and_proportion(filename,knwb)
            caps = whiteSpaceFeature(filename)
            digs = digitFeature(filename)
            links = linkFeature(filename)
            buzz = buzzwordFeature(filename)
            l = classify(filename,knwb,m,c)
            plt.scatter(prob,buzz, c= 'r' if l =="spam" else 'b', marker= 'x',s=50)

    for i in range(400):
        filename = "automarker/emails/ham"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            prob, bang = probability_and_proportion(filename,knwb)
            caps = whiteSpaceFeature(filename)
            digs = digitFeature(filename)
            links = linkFeature(filename)
            buzz = buzzwordFeature(filename)
            l = classify(filename,knwb,m,c)
            plt.scatter(prob,buzz, c= 'r' if l =="spam" else 'b', marker= 'D',s=50)

    xx = np.linspace(0,2,10)
    yy = xx*m+c
    plt.plot(xx,yy,"k-",linewidth=3)
    plt.xlim(0.7,1.4)
    plt.ylim(-10,100)
    plt.grid(True)
    val1 = (val1/n1)
    val2 = (val2/n1)
    val3 = val3/n2
    val4 = val4/n2
    plt.scatter(val1,val2,c='g',s=100)
    plt.scatter(val3,val4,c='g',s=100)
    plt.show()


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
                            knowledgebase[(w,'s')] = 1
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
        if (abs(pws-pwh)> max(pws,pwh)*0.30):
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

def testBoth():
    ACCSS = []
    ACCSL = []
    for i in range(20):
        train_ham, train_spam, test_hams, test_spam = divideTestTrain()
        ACCSL.append(getAccuracyFinal(train_ham, train_spam, test_hams, test_spam))
        ACCSS.append(getAccuracyNaivebayes(train_ham, train_spam, test_hams, test_spam))
    print "----Simple Naive Bayes---"
    #print "Accuracy per iteration: {}".format(ACCSS)
    print "Mean Accuracy {:.3f}\n".format(np.mean(ACCSS))

    print "----Linear Classifier---"
    #print "Accuracy per iteration: {}".format(ACCSL)
    print "Mean Accuracy {:.3f}".format(np.mean(ACCSL))

def tuneSpamicity():
    tuneAccs =[]
    plat = platform.system()
    xx = np.linspace(0,0.9,90)
    ccc = 0
    for x in xx:
        ACCS = []
        SPAMICITY = x
        for i in range(10):
            train_ham, train_spam, test_hams, test_spam = divideTestTrain()
            ACCS.append(getAccuracyFinal(train_ham, train_spam, test_hams, test_spam))
        tuneAccs.append(np.mean(ACCS))
        ccc+=1
        print ccc
    fig = plt.figure()
    plt.plot(xx,tuneAccs,"k-",linewidth=3)
    plt.xlim(0,0.9)
    plt.ylim(0,1)
    plt.grid(True)
    plt.show()

def tuneAmaount():
    tuneAccs =[]
    plat = platform.system()
    xx = np.arange(2,13)
    ccc = 0
    for x in xx:
        ACCS = []
        AMOUNT = x
        for i in range(10):
            train_ham, train_spam, test_hams, test_spam = divideTestTrain()
            ACCS.append(getAccuracyFinal(train_ham, train_spam, test_hams, test_spam))
        tuneAccs.append(np.mean(ACCS))
        ccc+=1
        print ccc
    fig = plt.figure()
    plt.plot(xx,tuneAccs,"k-",linewidth=3)
    plt.xlim(2,12)
    plt.ylim(0,1)
    plt.grid(True)
    plt.show()

def tunner2d():
    spamrange = np.arange(0,0.9,0.05)
    amountRange = np.arange(2,20,1)
    tuneAccs = np.zeros((18,18))
    tot_iters = 18*18
    percCounter =0
    xi = 0
    for x in spamrange:
        SPAMICITY = x
        for y in amountRange:
            ACCS = []
            AMOUNT = y
            for i in range(5):
                train_ham, train_spam, test_hams, test_spam = divideTestTrain()
                ACCS.append(getAccuracyFinal(train_ham, train_spam, test_hams, test_spam))

            tuneAccs[xi,y-2] = np.mean(ACCS)
            percCounter += 1
            percentCompleted = (percCounter/tot_iters) *100
            print "{:.2f}%".format(percentCompleted)
        xi +=1
    for i in range(5):
        xmax = tuneAccs.argmax(0)
        ymax = tuneAccs.argmax(1)
        print xmax
        print ymax
        print tuneAccs.max()
        tuneAccs[xmax,ymax] =0.0

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
