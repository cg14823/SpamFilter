# -*- coding: utf-8 -*-
from __future__ import division
import re
import sys
import cPickle as pickle
import os
import math
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import random
import operator

# 0 -> ham
# 1 -> spam

stopWords = ["", "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can\'t", "cannot", "could", "couldn\'t", "did", "didn\'t", "do", "does", "doesn\'t", "doing", "don\'t", "down", "during", "each", "few", "for", "from", "further", "had", "hadn\'t", "has", "hasn\'t", "have", "haven\'t", "having", "he", "he\'d", "he\'ll", "he\'s", "her", "here", "here\'s", "hers", "herself", "him", "himself", "his", "how", "how\'s", "i", "i\'d", "i\'ll", "i\'m", "i\'ve", "if", "in", "into", "is", "isn\'t", "it", "it\'s", "its", "itself", "let\'s", "me", "more", "most", "mustn\'t", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours","ourselves", "out", "over", "own", "same", "shan\'t", "she", "she\'d", "she\'ll", "she\'s", "should", "shouldn\'t", "so", "some", "such", "than", "that", "that\'s", "the", "their", "theirs", "them", "themselves", "then", "there", "there\'s", "these", "they", "they\'d", "they\'ll", "they\'re", "they\'ve", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn\'t", "we", "we\'d", "we\'ll", "we\'re", "we\'ve", "were", "weren\'t", "what", "what\'s", "when", "when\'s", "where", "where\'s", "which", "while", "who", "who\'s", "whom", "why", "why\'s", "with", "won\'t", "would", "wouldn\'t", "you", "you\'d", "you\'ll", "you\'re", "you\'ve", "your", "yours", "yourself", "yourselves", ""]

AMOUNT = 18
SPAMICITY = 0.58


def main():
    n = len(sys.argv)

    if n == 2 and sys.argv[1] == "-t":
        if os.path.isfile("knowledgebase.p"):
            os.remove("knowledgebase.p")
        knowledgebase = buildTrainingSet(range(400),range(100))
        with open("knowledgebase.p","wb") as handle:
            pickle.dump(knowledgebase, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if os.path.isfile("knnClassifier.p"):
            os.remove("knnClassifier.p")
        knn = createKNN(range(400),range(100),knowledgebase)
        with open("knnClassifier.p","wb") as handle:
            pickle.dump(knn, handle, protocol = pickle.HIGHEST_PROTOCOL)
    elif n == 2:
        if os.path.isfile("knowledgebase.p"):
            with open("knowledgebase.p","rb") as handle:
                knwb = pickle.load(handle)
                if os.path.isfile("knnClassifier.p"):
                    with open("knnClassifier.p","rb") as handle:
                        knn  = pickle.load(handle)
                        if os.path.isfile(sys.argv[1]):
                            print classify(sys.argv[1],knwb,knn) +"\n"
                        else:
                            print sys.argv[1] +" was not found!\n"
                else:
                    print "knnClassifier.p was not found!\n"
        else:
            print "knowledgebase.p was not found!\n"
    else:
        print "Invalid Parameters\n"

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

def linearPredict(p,m,c):
    liney = m*p[0]+c
    if (p[1] >liney):
        return "spam"
    return "ham"

def classify(filename,knwb,knn):
    ps, ratio = probability_and_proportion(filename,knwb)
    white = whiteSpaceFeature(filename)
    caps = capitalLetterFeature(filename)
    return knn.predict(np.matrix([ps,ratio,caps,white]))[0]

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
    elif (spamScore == hamScore):
        val = random.randint(0,1)
        if val == 0:
            return "spam"
        else:
            return "ham"
    else:
        return "ham"

def testAll(knn,knwb):
    ACCSIMPLE = []
    ACCIMBAYES = []
    TPRIMBAYES = []
    FPRIMBAYES = []
    ACCLINEAR = []
    ACCKNN = []
    TPRKNN = []
    FPRKNN = []
    ACCSVC = []
    TPRSVC = []
    FPRSVC = []
    ACCS15 = []
    TPRLINEAR = []
    TPRSIMPLE = []
    FPRLINEAR = []
    FPRSIMPLE = []
    for i in range(30):
        print str(i*100/30)+"%"
        train_ham, train_spam, test_hams, test_spam = divideTestTrain()
        ACC,TPR,FPR = getAccuracyFinal(train_ham, train_spam, test_hams, test_spam)
        ACCLINEAR.append(ACC)
        TPRLINEAR.append(TPR)
        FPRLINEAR.append(FPR)
        ACC,TPR,FPR = getAccuracyNaivebayes(train_ham, train_spam, test_hams, test_spam)
        ACCSIMPLE.append(ACC)
        TPRSIMPLE.append(TPR)
        FPRSIMPLE.append(FPR)

        #knwb = buildTrainingSet(train_ham,train_spam)
        # KNN Classifier, SVC, improved Bayes
        #knn = createKNN(train_ham, train_spam,knwb,w='distance',n =5)
        svc = createSVC(train_ham, train_spam,knwb)

        hamsPredict = []
        spamsPredict = []
        hamsPredictS = []
        spamsPredictS = []
        hamsPredictB = []
        spamsPredictB = []
        hamsPredict15 = []
        spamsPredict15 = []
        for i in test_hams:
            filename = "public/ham"+("%03d"%i)+".txt"
            prob, ratio = probability_and_proportion(filename,knwb)
            caps = capitalLetterFeature(filename)
            white = whiteSpaceFeature(filename)
            #digs = digitFeature(filename)
            #links = linkFeature(filename)
            hamsPredict.append(knn.predict(np.matrix([prob,ratio,caps,white])))
            hamsPredictS.append(svcPredict(svc,filename,knwb))
            hamsPredictB.append(simpleBayes(knwb,filename))
            hamsPredict15.append(testTopTen(filename,knwb))

        for i in test_spam:
            filename = "public/spam"+("%03d"%i)+".txt"
            prob, ratio = probability_and_proportion(filename,knwb)
            caps = capitalLetterFeature(filename)
            white = whiteSpaceFeature(filename)
            #digs = digitFeature(filename)
            #links = linkFeature(filename)
            spamsPredict.append(knn.predict(np.matrix([prob,ratio,caps,white])))
            spamsPredictS.append(svcPredict(svc,filename,knwb))
            spamsPredictB.append(simpleBayes(knwb,filename))
            spamsPredict15.append(testTopTen(filename,knwb))

        TP = spamsPredict.count("spam")
        TN = hamsPredict.count("ham")
        FP = spamsPredict.count("ham")
        FN = hamsPredict.count("spam")
        ACC = (TP + TN)/(TP + TN + FP + FN)
        ACCKNN.append(ACC)
        try:
            TPR = TP/(TP+FN)
            FPR = FP/(FP+TN)
            TPRKNN.append(TPR)
            FPRKNN.append(FPR)
        except ZeroDivisionError:
            print "vals was 0\n"

        TP = spamsPredictS.count("spam")
        TN = hamsPredictS.count("ham")
        FP = spamsPredictS.count("ham")
        FN = hamsPredictS.count("spam")
        ACC = (TP + TN)/(TP + TN + FP + FN)
        ACCSVC.append(ACC)
        try:
            TPR = TP/(TP+FN)
            FPR = FP/(FP+TN)
            TPRSVC.append(TPR)
            FPRSVC.append(FPR)
        except ZeroDivisionError:
            print "vals was 0\n"


        TP = spamsPredictB.count("spam")
        TN = hamsPredictB.count("ham")
        FP = spamsPredictB.count("ham")
        FN = hamsPredictB.count("spam")
        ACC = (TP + TN)/(TP + TN + FP + FN)
        ACCIMBAYES.append(ACC)
        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
        TPRIMBAYES.append(TPR)
        FPRIMBAYES.append(FPR)
        TP = spamsPredict15.count("spam")
        TN = hamsPredict15.count("ham")
        FP = spamsPredict15.count("ham")
        FN = hamsPredict15.count("spam")
        ACC = (TP + TN)/(TP + TN + FP + FN)
        ACCS15.append(ACC)


    print "----Simple Naive Bayes---"
    #print "Accuracy per iteration: {}".format(ACCSS)
    print "Mean Accuracy {:.3f}".format(np.mean(ACCSIMPLE))
    print "TPR = {0}, FPR = {1}".format(np.mean(TPRSIMPLE),np.mean(FPRSIMPLE))
    print "----Improved Bayes---"
    #print "Accuracy per iteration: {}".format(ACCSL)
    print "Mean Accuracy {:.3f}".format(np.mean(ACCIMBAYES))
    print "TPR = {0}, FPR = {1}".format(np.mean(TPRIMBAYES),np.mean(FPRIMBAYES))
    print "----Top 15---"
    #print "Accuracy per iteration: {}".format(ACCSL)
    print "Mean Accuracy {:.3f}".format(np.mean(ACCS15))
    print "----Linear Classifier---"
    #print "Accuracy per iteration: {}".format(ACCSL)
    print "Mean Accuracy {:.3f}".format(np.mean(ACCLINEAR))
    print "TPR = {0}, FPR = {1}".format(np.mean(TPRLINEAR),np.mean(FPRLINEAR))

    print "----KNN Classifier---"
    #print "Accuracy per iteration: {}".format(ACCSL)
    print "Mean Accuracy {:.3f}".format(np.mean(ACCKNN))
    print "TPR = {0}, FPR = {1}".format(np.mean(TPRKNN),np.mean(FPRKNN))
    print "----SVC Classifier---"
    #print "Accuracy per iteration: {}".format(ACCSL)
    print "Mean Accuracy {:.3f}".format(np.mean(ACCSVC))
    print "TPR = {0}, FPR = {1}".format(np.mean(TPRSVC),np.mean(FPRSVC))

    print "\nNum Of Stopwords : {0}".format(len(stopWords))

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
    hF3 = []
    hF4 = []
    #hF5 = []
    sF1 = []
    sF2 = []
    sF3 = []
    sF4 = []
    #sF5 = []

    for i in train_ham:
        filename = "public/ham"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            prob, ratio = probability_and_proportion(filename,knwb)
            caps = capitalLetterFeature(filename)
            white = whiteSpaceFeature(filename)
            #digs = digitFeature(filename)
            #links = linkFeature(filename)
            if (ratio >= 0):
                hF1.append(prob)
                hF2.append(ratio)
                hF3.append(caps)
                hF4.append(white)
                #hF5.append(links)

    for i in train_spam:
        filename = "public/spam"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            prob, ratio = probability_and_proportion(filename,knwb)
            caps = capitalLetterFeature(filename)
            white = whiteSpaceFeature(filename)
            #digs = digitFeature(filename)
            #links = linkFeature(filename)
            if (ratio >= 0):
                sF1.append(prob)
                sF2.append(ratio)
                sF3.append(caps)
                sF4.append(white)
                #sF5.append(links)
    labels = ["ham"]*len(hF1) +["spam"]*len(sF1)
    f1 = np.append(hF1,sF1)
    f2 = np.append(hF2,sF2)
    f3 = np.append(hF3,sF3)
    f4 = np.append(hF4,sF4)
    #f5 = np.append(hF5,sF5)
    X = np.zeros((len(f1),4))
    for i in range(len(f1)):
        X[i,0] = f1[i]
        X[i,1] = f2[i]
        X[i,2] = f3[i]
        X[i,3] = f4[i]
        #X[i,3] = f5[i]


    svc = SVC(kernel=k)
    svc.fit(X,labels)
    return svc

def svcPredict (svc, filename, knwb):
    prob, ratio = probability_and_proportion(filename,knwb)
    caps = capitalLetterFeature(filename)
    white = whiteSpaceFeature(filename)
    #digs = digitFeature(filename)
    #links = linkFeature(filename)
    if ratio >= 0:
        return svc.predict(np.matrix([prob,ratio,caps,white]))
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
    hF3 = []
    hF4 = []
    #hF5 = []
    sF1 = []
    sF2 = []
    sF3 = []
    sF4 = []
    #sF5 = []

    for i in train_ham:
        filename = "public/ham"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            prob, ratio = probability_and_proportion(filename,knwb)
            caps = capitalLetterFeature(filename)
            white = whiteSpaceFeature(filename)
            #digs = digitFeature(filename)
            #links = linkFeature(filename)
            hF1.append(prob)
            hF2.append(ratio)
            hF3.append(caps)
            hF4.append(white)
            #hF5.append(links)

    for i in train_spam:
        filename = "public/spam"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            prob, ratio = probability_and_proportion(filename,knwb)
            caps = capitalLetterFeature(filename)
            white = whiteSpaceFeature(filename)
            #digs = digitFeature(filename)
            #links = linkFeature(filename)
            syms = testsymtoletSingle(filename)
            sF1.append(prob)
            sF2.append(ratio)
            sF3.append(caps)
            sF4.append(white)
            #sF5.append(links)

    labels = ["ham"]*len(hF1) +["spam"]*len(sF1)
    f1 = np.append(hF1,sF1)
    f2 = np.append(hF2,sF2)
    f3 = np.append(hF3,sF3)
    f4 = np.append(hF4,sF4)
    #f5 = np.append(hF5,sF5)
    X = np.zeros((len(f1),4))
    for i in range(len(f1)):
        X[i,0] = f1[i]
        X[i,1] = f2[i]
        X[i,2] = f3[i]
        X[i,3] = f4[i]
        #X[i,3] = f5[i]

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
                    spamDict[w] = knwb[(w,'s')]
                else:
                    unkown += 1
    rFile.close()

    sortedHam = sorted(hamDict.items(),key=operator.itemgetter(1))
    sortedHam.reverse()
    sortedSpam = sorted(spamDict.items(),key=operator.itemgetter(1))
    sortedSpam.reverse()
    sortedHam =sortedHam[:15]
    sortedSpam = sortedSpam[:15]

    hamScore = 0.0
    spamScore = 0.0
    for i in range(len(sortedHam)):
        hamScore += math.log(sortedHam[i][1])
        spamScore += math.log(sortedSpam[i][1])
    if hamScore > spamScore:
        return "ham"
    else:
        return "spam"

main()
