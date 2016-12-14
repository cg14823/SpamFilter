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
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
import random
import platform
import operator

# 0 -> ham
# 1 -> spam

AMOUNT = 2
SPAMICITY = 0.00
stopWords = ['', 'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours\tourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves', '']
def main():
    n = len(sys.argv)

    if n == 2 and sys.argv[1] == "-t":

        if os.path.isfile("knowledgebase.p"):
            os.remove("knowledgebase.p")
        knowledgebase = buildTrainingSet(range(400),range(100))
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
    elif n ==2  and sys.argv[1] == "-t10":
        with open("knowledgebase.p","rb") as handle:
            knwb = pickle.load(handle)
        testTopTen("public\ham000.txt",knwb)
    elif n ==2  and sys.argv[1] == "-ca":
        testMultipleComplexBayes()
    elif n ==2  and sys.argv[1] == "-tuneS":
        tuneSpamicity()
    elif n ==2  and sys.argv[1] == "-tuneA":
        tuneAmaount()
    elif n ==2  and sys.argv[1] == "-tune2d":
        tunner2d()
    elif n ==2  and sys.argv[1] == "-testSVC":
        testSVC(20)
    elif n ==2  and sys.argv[1] == "-testKNN":
        testKNN(20)
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
        symbolCount = len(re.sub(r'[A-za-z]',"",nospaces))
        characterCount = allcount-symbolCount
        prop = -0.5
        try:
            prop = symbolCount/characterCount
        except ZeroDivisionError:
            print "cgharactercount was 0\n"
        return prop
    return -0.5

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

    for i in nham:
        filename = "public/ham"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            rFile = open(filename,'r' )
            for line in rFile:
                nline = re.sub(r'[^A-Za-z ]',"",line)
                words = nline.split(' ')
                for w in words:
                    if w != ""and not (w.lower() in stopWords):
                        if (w,'h') in knowledgebase:
                            knowledgebase[(w,'h')] +=1
                        else:
                            knowledgebase[(w,'h')] = 1
                        wordsHam +=1

    for i in nspam:
        filename = "public/spam"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            rFile = open(filename,'r' )
            for line in rFile:
                nline = re.sub(r'[^A-Za-z ]',"",line)
                words = nline.split(' ')
                for w in words:
                    if w != "" and not (w.lower() in stopWords):
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
        if knowledgebase[(key[0],'s')] +knowledgebase[(key[0],'h')] < AMOUNT:
            continue

        pws = knowledgebase[(key[0],'s')] / (wordsSpam + totWords)
        pwh = knowledgebase[(key[0],'h')] /(wordsHam +totWords)

        # ignore words with similar probabilities for spam and ham
        if (abs(pws-pwh) / (pws+pwh) > SPAMICITY):
            knwb[key[0],'s'] = pws
            knwb[key[0],'h'] = pwh
    return knwb

def buildTrainingSetSimple(nham,nspam):
    knowledgebase = dict()
    wordsHam =0
    wordsSpam =0
    hamsProb = []
    hamsbang = []
    spamsProb = []
    spamsbang = []

    for i in nham:
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

    for i in nspam:
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


        pws = knowledgebase[(key[0],'s')] / (wordsSpam + totWords)
        pwh = knowledgebase[(key[0],'h')] /(wordsHam +totWords)

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

def probability_and_proportion(filename, knwb):
    # getting words in email
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
                #else:
                #    spamScore += math.log(0.99)
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
    knwb = buildTrainingSetSimple(trainsH,trainS)
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
    knwb = buildTrainingSet(trainsH,trainS)
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
    ACCSIMPLE = []
    ACCIMBAYES = []
    ACCLINEAR = []
    ACCKNN = []
    ACCSVC = []
    for i in range(25):
        train_ham, train_spam, test_hams, test_spam = divideTestTrain()
        ACCLINEAR.append(getAccuracyFinal(train_ham, train_spam, test_hams, test_spam))
        ACCSIMPLE.append(getAccuracyNaivebayes(train_ham, train_spam, test_hams, test_spam))
        knwb = buildTrainingSet(train_ham,train_spam)
        # KNN Classifier, SVC, improved Bayes
        knn = createKNN(train_ham, train_spam,knwb,w='distance',n =5)
        svc = createSVC(train_ham, train_spam,knwb)

        hamsPredict = []
        spamsPredict = []
        hamsPredictS = []
        spamsPredictS = []
        hamsPredictB = []
        spamsPredictB = []
        for i in test_hams:
            filename = "public/ham"+("%03d"%i)+".txt"
            prob, ratio = probability_and_proportion(filename,knwb)
            hamsPredict.append(knn.predict(np.matrix([prob,ratio])))
            hamsPredictS.append(svcPredict(svc,filename,knwb))
            hamsPredictB =simpleBayes(knwb,filename)

        for i in test_spam:
            filename = "public/spam"+("%03d"%i)+".txt"
            prob, ratio = probability_and_proportion(filename,knwb)
            spamsPredict.append(knn.predict(np.matrix([prob,ratio])))
            spamsPredictS.append(svcPredict(svc,filename,knwb))
            spamsPredictB =simpleBayes(knwb,filename)

        TP = spamsPredict.count("spam")
        TN = hamsPredict.count("ham")
        FP = spamsPredict.count("ham")
        FN = hamsPredict.count("spam")
        ACC = (TP + TN)/(TP + TN + FP + FN)
        ACCKNN.append(ACC)

        TP = spamsPredictS.count("spam")
        TN = hamsPredictS.count("ham")
        FP = spamsPredictS.count("ham")
        FN = hamsPredictS.count("spam")
        ACC = (TP + TN)/(TP + TN + FP + FN)
        ACCSVC.append(ACC)

        TP = spamsPredictB.count("spam")
        TN = hamsPredictB.count("ham")
        FP = spamsPredictB.count("ham")
        FN = hamsPredictB.count("spam")
        ACC = (TP + TN)/(TP + TN + FP + FN)
        ACCIMBAYES.append(ACC)

    print "----Simple Naive Bayes---"
    #print "Accuracy per iteration: {}".format(ACCSS)
    print "Mean Accuracy {:.3f}".format(np.mean(ACCSIMPLE))
    print "----Improved Bayes---"
    #print "Accuracy per iteration: {}".format(ACCSL)
    print "Mean Accuracy {:.3f}".format(np.mean(ACCIMBAYES))
    print "----Linear Classifier---"
    #print "Accuracy per iteration: {}".format(ACCSL)
    print "Mean Accuracy {:.3f}".format(np.mean(ACCLINEAR))
    print "----KNN Classifier---"
    #print "Accuracy per iteration: {}".format(ACCSL)
    print "Mean Accuracy {:.3f}".format(np.mean(ACCKNN))
    print "----SVC Classifier---"
    #print "Accuracy per iteration: {}".format(ACCSL)
    print "Mean Accuracy {:.3f}".format(np.mean(ACCSVC))

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

def createSVC(train_ham, train_spam,knwb, k="rbf"):
    hF1 = []
    hF2 = []
    sF1 = []
    sF2 = []

    for i in train_ham:
        filename = "public/ham"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            prob, ratio = probability_and_proportion(filename,knwb)
            if (ratio >= 0):
                hF1.append(prob)
                hF2.append(ratio)

    for i in train_spam:
        filename = "public/spam"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            prob, ratio = probability_and_proportion(filename,knwb)
            if (ratio >= 0):
                sF1.append(prob)
                sF2.append(ratio)
    labels = ["ham"]*len(hF1) +["spam"]*len(sF1)
    f1 = np.append(hF1,sF1)
    f2 = np.append(hF2,sF2)
    X = np.zeros((len(f1),2))
    for i in range(len(f1)):
        X[i,0] = f1[i]
        X[i,1] = f2[i]


    svc = SVC(kernel=k)
    svc.fit(X,labels)
    return svc

def svcPredict (svc, filename, knwb):
    prob, ratio = probability_and_proportion(filename,knwb)
    if ratio >= 0:
        return svc.predict(np.matrix([prob,ratio]))
    elif prob > 1:
        return "spam"
    else:
        return "ham"

def testSVC(its = 5):
    ACCS = []
    for i in range(its):
        train_ham, train_spam, test_hams, test_spam = divideTestTrain()
        knwb = buildTrainingSet(train_ham,train_spam)
        svc = createSVC(train_ham, train_spam,knwb)

        hamsPredict = []
        spamsPredict = []
        for i in test_hams:
            filename = "public/ham"+("%03d"%i)+".txt"
            hamsPredict.append(svcPredict(svc,filename,knwb))
        for i in test_spam:
            filename = "public/spam"+("%03d"%i)+".txt"
            spamsPredict.append(svcPredict(svc,filename,knwb))

        TP = spamsPredict.count("spam")
        TN = hamsPredict.count("ham")
        FP = spamsPredict.count("ham")
        FN = hamsPredict.count("spam")
        ACC = (TP + TN)/(TP + TN + FP + FN)
        ACCS.append(ACC)
        print ACC

    print "Accuracy per iteration: {}".format(ACCS)
    print "Mean Accuracy {}".format(np.mean(ACCS))

def createKNN(train_ham, train_spam, knwb, w ='uniform', n = 3):
    hF1 = []
    hF2 = []
    sF1 = []
    sF2 = []

    for i in train_ham:
        filename = "public/ham"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            prob, ratio = probability_and_proportion(filename,knwb)
            hF1.append(prob)
            hF2.append(ratio)

    for i in train_spam:
        filename = "public/spam"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            prob, ratio = probability_and_proportion(filename,knwb)
            sF1.append(prob)
            sF2.append(ratio)

    labels = ["ham"]*len(hF1) +["spam"]*len(sF1)
    f1 = np.append(hF1,sF1)
    f2 = np.append(hF2,sF2)
    X = np.zeros((len(f1),2))
    for i in range(len(f1)):
        X[i,0] = f1[i]
        X[i,1] = f2[i]


    knn  = KNeighborsClassifier(weights=w, n_neighbors=n)
    knn.fit(X,labels)
    return knn

def testKNN(its = 5):
    ACCS = []
    for i in range(its):
        train_ham, train_spam, test_hams, test_spam = divideTestTrain()
        knwb = buildTrainingSet(train_ham,train_spam)
        knn = createKNN(train_ham, train_spam,knwb)

        hamsPredict = []
        spamsPredict = []
        for i in test_hams:
            filename = "public/ham"+("%03d"%i)+".txt"
            prob, ratio = probability_and_proportion(filename,knwb)
            hamsPredict.append(knn.predict(np.matrix([prob,ratio])))
        for i in test_spam:
            filename = "public/spam"+("%03d"%i)+".txt"
            prob, ratio = probability_and_proportion(filename,knwb)
            spamsPredict.append(knn.predict(np.matrix([prob,ratio])))

        TP = spamsPredict.count("spam")
        TN = hamsPredict.count("ham")
        FP = spamsPredict.count("ham")
        FN = hamsPredict.count("spam")
        ACC = (TP + TN)/(TP + TN + FP + FN)
        ACCS.append(ACC)

    print "Accuracy per iteration: {}".format(ACCS)
    print "Mean Accuracy {}".format(np.mean(ACCS))

def testTopTen(filename, knwb):
    hamDict = dict()
    spamDict =dict()
    unkown = 0
    rFile = open(filename,'r' )
    for line in rFile:
        nline = re.sub(r'[^A-Za-z ]',"",line)
        words = nline.split(' ')
        for w in words:
            if w != "":
                if (w,'s') in knwb:
                    hamDict[w] = knwb[(w,'h')]
                    spamDict[w] = knwb[(w,'h')]
                else:
                    unkown += 1
    rFile.close()

    sortedHam = sorted(hamDict.items(),key=operator.itemgetter(1))
    sortedHam.reverse()
    sortedSpam = sorted(spamDict.items(),key=operator.itemgetter(1))
    sortedSpam.reverse()
    print sortedHam[:10]
    print sortedSpam[:10]



main()
