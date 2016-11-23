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
    hsymbols = 0.0
    ssymbols = 0.0
    hletters = 0.0
    sletters = 0.0

    # Laplace correction added
    for i in range(nham):
        subjectLineReached = False
        filename = "public/ham"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            rFile = open(filename,'r' )
            emailWord = []
            for line in rFile:
                if not (subjectLineReached):
                    if "Subject:" in line:
                        subjectLineReached = True
                        line = line[9:]
                        line = re.sub('[\s]',"",line)
                        hletters += len(line)
                        line=re.sub('[A-Za-z1-9]',"",line)
                        hsymbols += len(line)
                else:
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
            if not subjectLineReached:
                rFile = open(filename,'r')
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
        subjectLineReached = False
        filename = "public/spam"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            rFile = open(filename,'r' )
            emailWord = []
            for line in rFile:
                if not (subjectLineReached):
                    if "Subject:" in line:
                        subjectLineReached = True
                        line = line[9:]
                        line = re.sub('[\s]',"",line)
                        sletters += len(line)
                        line=re.sub('[A-Za-z1-9]',"",line)
                        ssymbols += len(line)
                else:
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
            if not subjectLineReached:
                rFile = open(filename,'r')
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
        if knowledgebase[key[0],0] +knowledgebase[key[0],1] < 7:
            continue

        if key[1] == 1:
            pws = (knowledgebase[key] / nspam)
            knwb[key] = pws
        else:
            pwh = (knowledgebase[key] / nham)
            knwb[key] = pwh

    #print "HAM CAP PROP %f" % ((hcapwords/hwordstot)/(nham-2))
    #print "SPAM CAP PROP %f" % ((scapwords/swordstot)/(nspam-2))
    #print "Ham Symbols prop %f" % ((hsymbols/hletters)/(nham-2))
    #print "Spam Symbols prop %f" % ((ssymbols/sletters)/(nspam-2))
    return knwb

def testData(file, knwb):
    # getting words in email
    wordstot = 0.0
    capwords = 0.0
    letters = 0.0
    symbols = 0.0
    rFile = open(file,'r' )
    emailWord = []
    subjectLineReached = False
    for line in rFile:
        if not (subjectLineReached):
            if "Subject:" in line:
                subjectLineReached = True
                line = line[9:]
                line = re.sub('[\s]',"",line)
                letters += len(line)
                line=re.sub('[A-Za-z1-9]',"",line)
                symbols += len(line)
        else:
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
    if not subjectLineReached:
        rFile = open(file,'r' )
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
    # calculate ln(P(S|emailWord0 ... emailWordN) / P(S|emailWord0 ... emailWordN)) >1
    scorespam= 0
    scoreham = 0
    for w in emailWord:
        if (w,0) in knwb:
            scorespam += math.log(knwb[(w,1)])
            scoreham += math.log(knwb[(w,0)])


    scoreham  += math.log(4/5)
    scorespam += math.log(1/5)
    #print "scoreham %f" % scoreham
    #print "scorespam %f" % scorespam
    if scoreham >= scorespam:
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
            print counter
            counter +=1
            hams.append(testData(filename,knwb))

    for i in range(nspam):
        filename = "public/spam"+("%03d"%i)+".txt"
        if os.path.isfile(filename):
            print i
            spams.append(testData(filename,knwb))

    for i in range(len(hams)):
        if("spam" in hams[i]):
            print i
            hamswrong +=1
    print "hams wrong %d"% hamswrong

    print "\n spams wrong %d\n",len(spams)

    while ("ham" in spams):
        spamswrong += 1
        print str(spams.index("ham"))+ " "
        spams.remove("ham")

    print "spams wrong %d"%spamswrong

main()
