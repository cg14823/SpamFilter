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

def getBayes(nham,nspam, countDic):
    # ham =0,spam=1
    # for every dic(w,spam) = P(S|W)
    knwb = dict()
    for key in countDic:
        countDic[key]
        if key[1] == 1:
            pwsXps = countDic[key] *nspam
            psgivenw = pwsXps / (pwsXps+(countDic[(key[0],0)] * nham))
            knwb[key] = psgivenw
        else:
            pwhXph = countDic[key] *nham
            phgivenw = pwhXph / (pwhXph+(countDic[(key[0],1)] * nspam))
            knwb[key] = psgivenw

    return knwb


def buildTrainingSet(nham,nspam):
    knowledgebase = dict()
    swordstot = 0.0
    scapwords = 0.0
    hwordstot = 0.0
    hcapwords = 0.0
    # Laplace correction added
    for i in range(nham):
        filename = "public/ham"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            rFile = open(filename,'r' )
            emailWord = []
            for line in rFile:
                nline = re.sub(r'[^A-Za-z ]',"",line)
                nline = re.sub('\s+'," ",nline)
                words = nline.split(' ')
                for w in words:
                    if w == "":
                        continue
                    hwordstot += 1
                    if w.isupper():
                        hcapwords += 1
                    if w in neutral_words:
                        continue
                    if not(w in emailWord):
                        if (w,0) in knowledgebase:
                            knowledgebase[(w,0)] += 1
                        else:
                            knowledgebase[(w,0)] = 2
                            knowledgebase[(w,1)] = 1
                        emailWord.append(w)
            rFile.close()

    for i in range(nspam):
        filename = "public/spam"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            rFile = open(filename,'r' )
            emailWord = []
            for line in rFile:
                nline = re.sub(r'[^A-Za-z ]',"",line)
                words = nline.split(' ')
                for w in words:
                    if w == "":
                        continue
                    swordstot += 1
                    if w.isupper():
                        scapwords += 1
                    if w in neutral_words:
                        continue
                    if not(w in emailWord):
                        if (w,1) in knowledgebase:
                            knowledgebase[(w,1)] += 1
                        else:
                            knowledgebase[(w,1)] = 2

                        if not ((w,0) in knowledgebase):
                            knowledgebase[(w,0)] = 1
                        emailWord.append(w)
            rFile.close()

    knwb = dict()
    nham +=2
    nspam +=2
    for key in knowledgebase:
        if key[1] == 1:
            pws = (knowledgebase[key] / nspam)
            pwh = knowledgebase[(key[0],0)] / nham
            if ( abs(pws-pwh) > 0.05):
                pwsXps = (knowledgebase[key] / nspam) * (nspam/(nspam+nham))
                psgivenw = pwsXps / (pwsXps+(knowledgebase[(key[0],0)] *(nham/(nspam+nham))))
                knwb[key] = psgivenw
        else:
            pwh = (knowledgebase[key] / nham)
            pws = knowledgebase[(key[0],1)] / nspam
            if ( abs(pws-pwh) > 0.05 ):
                pwhXph = (knowledgebase[key] / nham) * (nham/(nspam+nham))
                phgivenw = pwhXph / (pwhXph+(knowledgebase[(key[0],1)] * (nspam/(nspam+nham))))
                knwb[key] = psgivenw
    print "HAM CAP PROP %f" % (hcapwords/hwordstot)
    print "SPAM CAP PROP %f" % (scapwords/swordstot)
    return knwb

def testData(file, knwb):
    # getting words in email
    wordstot = 0.0
    capwords = 0.0
    rFile = open(file,'r' )
    emailWord = []
    for line in rFile:
        nline = re.sub(r'[^A-Za-z ]',"",line)
        words = nline.split(' ')
        for w in words:
            if w == "":
                continue
            wordstot += 1
            if w.isupper():
                capwords += 1
            if not(w in emailWord):
                emailWord.append(w)
    rFile.close()

    # calculate ln(P(S|emailWord0 ... emailWordN) / P(S|emailWord0 ... emailWordN)) >1
    logpsw = 0
    logphw = 0
    for w in emailWord:
        if (w,0) in knwb:
            logpsw += math.log(knwb[(w,1)])
            logphw += math.log(knwb[(w,0)])
    #print logphw-logpsw
    if logphw - logpsw - (capwords/wordstot)*10>= 0:
        return "ham"
    else:
        return "spam"

def testAll(knwb,nham,nspam):
    hams = []
    spams =[]
    for i in range(nham):
        filename = "public/ham"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            hams.append(testData(filename,knwb))

    for i in range(nspam):
        filename = "public/spam"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            spams.append(testData(filename,knwb))

    while ("spam" in hams):
        print str(hams.index("spam"))+ " "
        hams.remove("spam")

    print "\n spams wrong\n"

    while ("ham" in spams):
        print str(spams.index("ham"))+ " "
        spams.remove("ham")

main()