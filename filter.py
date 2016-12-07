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

# 0 -> ham
# 1 -> spam
neutral_words = ["the","The","be","Be","to","To","Subject","in","In","a","A",
    "an","and","of","Of","I","at","At","by","By","this","This","because","Because",
    "or","Or","there","There","their","Their","also","for","For","are","Are","so","So",
    "Too","too","is","Is","as","As","it","It"]



def main():
    n = len(sys.argv)
    if n == 2 and sys.argv[1] == "-t":
        if os.path.isfile("knowledgebase.p"):
            os.remove("knowledgebase.p")
        knowledgebase = buildTrainingSet(400,100)
        print len(knowledgebase)
        with open("knowledgebase.p","wb") as handle:
            pickle.dump(knowledgebase, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif n ==2  and sys.argv[1] == "-tAll":
        with open("knowledgebase.p","rb") as handle:
            knwb = pickle.load(handle)
        testAll(knwb,400,100)
    elif n==2 and sys.argv[1] == "-bang":
        with open("knowledgebase.p","rb") as handle:
            knwb = pickle.load(handle)
        testAll(knwb,400,100)
    elif n == 2:
        with open("knowledgebase.p","rb") as handle:
            knwb = pickle.load(handle)
        if os.path.isfile(sys.argv[1]):
            print testData(sys.argv[1],knwb) +"\n"
        else:
            print sys.argv[1] +" was not found!\n"
    else:
        print "Usage: -t    Build Training Set\n"


def testBang(nham,nspam):
    knowledgebase = dict()
    wordsHam = 0
    wordsSpam = 0
    bangCount = 0

    for i in range(nham):
        filename = "public/ham"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            rFile = open(filename,'r' )
            for line in rFile:
                nline = re.sub(r'[^A-Za-z! ]',"",line)
                words = nline.split(' ')
                for w in words:
                    if w != "":
                        bangCount += w.count("!")
                        wordsHam += 1
    wordsHam = bangCount/wordsHam
    bangCount = 0
    for i in range(nspam):
        filename = "public/spam"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            rFile = open(filename,'r' )
            for line in rFile:
                nline = re.sub(r'[^A-Za-z! ]',"",line)
                words = nline.split(' ')
                for w in words:
                    if w != "":
                        bangCount += w.count("!")
                        wordsSpam += 1
    wordsSpam = bangCount/wordsSpam
    bangCount = 0

    print "Ham : {0}\tSpam : {1}".format(wordsHam,wordsSpam)


def testLinkSingle(filename):
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
        if(prop >= 0.5):
            print filename
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
        if (abs(pws-pwh)/(pws+pwh) > 0.1):
            knwb[key[0],'s'] = pws
            knwb[key[0],'h'] = pwh

    #print "HAM CAP PROP %f" % ((hcapwords/hwordstot)/(nham-2))
    #print "SPAM CAP PROP %f" % ((scapwords/swordstot)/(nspam-2))
    #print "Ham Symbols prop %f" % ((hsymbols/hletters)/(nham-2))
    #print "Spam Symbols prop %f" % ((ssymbols/sletters)/(nspam-2))
    return knwb


def testPlot(file, knwb):
    # getting words in email
    spamScore =0
    hamScore =0
    rFile = open(file,'r' )
    bangCount = testLinkSingle(file)
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

    return hamScore/spamScore , bangCount

def testData(file, knwb):
    # getting words in email
    spamScore =0
    hamScore =0
    rFile = open(file,'r' )
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
    if hamScore >= spamScore:
        return "ham"
    else:
        return "spam"


def testAll(knwb,nham,nspam):
    hamsProb = []
    hamsbang = []
    spamsProb = []
    spamsbang = []
    counter =0
    spamswrong = 0
    hamswrong = 0

    fig = plt.figure()
    for i in range(nham):
        filename = "public/ham"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            prob, bang = testPlot(filename,knwb)
            plt.scatter(prob,bang,c ='b')
            if (bang >=0):
                hamsProb.append(prob)
                hamsbang.append(bang)


    for i in range(nspam):
        filename = "public/spam"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            prob, bang = testPlot(filename,knwb)
            plt.scatter(prob,bang,c ='r')
            if (bang >=0):
                spamsProb.append(prob)
                spamsbang.append(bang)

    center1 = [np.mean(hamsProb),np.mean(hamsbang)]
    center2 =[np.mean(spamsProb),np.mean(spamsbang)]
    m,c = findLinearBoundry(center1,center2)



    ps =[]
    labels = []
    for i in range(nham):
        filename = "automarker/emails/ham"+("%03d"%i)+".txt"
        filename1 = "automarker/emails/spam"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            prob, bang = testPlot(filename,knwb)
            p =[prob,bang]
            ps.append(p)
            if (bang >=0):
                l = linearPredict([prob,bang],m,c)
                labels.append(l)
            else:
                l= "spam" if prob >= 1 else "ham"
                labels.append(l)


        if os.path.isfile(filename1):
            prob, bang = testPlot(filename1,knwb)
            p =[prob,bang]
            ps.append(p)
            if (bang >=0):
                l = linearPredict([prob,bang],m,c)
                labels.append(l)
            else:
                l= "spam" if prob >= 1 else "ham"
                labels.append(l)

    for i in range(len(ps)):
        plt.scatter(ps[i][0],ps[i][1],s=50,marker='D',c='r'if labels[i] == "spam" else 'b')


    xx = np.linspace(0,2,100)
    yy = m*xx+c
    plt.xlim(0,2)
    plt.ylim(-1,2)
    plt.plot(xx,yy,"k-",linewidth = 3)
    plt.show()

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
main()
