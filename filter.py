from __future__ import division
import re
import sys
import cPickle as pickle
import os
import math

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
    elif n == 2:
        with open("knowledgebase.p","rb") as handle:
            knwb = pickle.load(handle)
        if os.path.isfile(sys.argv[1]):
            print testData(sys.argv[1],knwb) +"\n"
        else:
            print sys.argv[1] +" was not found!\n"
    else:
        print "Usage: -t    Build Training Set\n"


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
    hams = []
    spams =[]
    counter =0
    spamswrong = 0
    hamswrong = 0
    for i in range(nham):
        filename = "public/ham"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            access = testData(filename,knwb)
            if (access == "spam"):
                print "Error at ham %d" %i
            hams.append(access)

    for i in range(nspam):
        filename = "public/spam"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            access = testData(filename,knwb)
            if (access == "ham"):
                print "Error at spam %d" %i
            spams.append(access)

main()
