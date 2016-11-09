import re
import sys
import cPickle as pickle
import os
import math
# 0 -> ham
# 1 -> spam
def main():
    n = len(sys.argv)
    print sys.argv
    if n == 2 and sys.argv[1] == "-t":
        os.remove("knowledgebase.p")
        knowledgebase = buildTrainingSet(400,100)
        pickle.dump(knowledgebase, open("knowledgebase.p","wb"))
    elif n == 2:
        knwb = pickle.load(open("knowledgebase.p","rb"))
        testData(sys.argv[1],knwb)
    else:
        print "Usage: -t    Build Training Set\n"

def buildTrainingSet(nham,nspam):
    knowledgebase = dict()

    for i in range(nham):
        filename = "public/ham"+("%03d"%i)+".txt"
        rFile = open(filename,'r' )
        emailWord = []
        for line in rFile:
            nline = re.sub(r'[^A-Za-z ]',"",line)
            nline = re.sub('\s+'," ",nline)
            words = nline.split(' ')
            for w in words:
                if w == "":
                    continue
                if not(w in emailWord):
                    if (w,0) in knowledgebase:
                        knowledgebase[(w,0)] += 1
                    else:
                        knowledgebase[(w,0)] = 1
                        knowledgebase[(w,1)] = 1
                    emailWord.append(w)
        rFile.close()

    for i in range(nspam):
        filename = "public/spam"+("%03d"%i)+".txt"
        rFile = open(filename,'r' )
        emailWord = []
        for line in rFile:
            nline = re.sub(r'[^A-Za-z ]',"",line)
            words = nline.split(' ')
            for w in words:
                if w == "":
                    continue
                if not(w in emailWord):
                    if (w,1) in knowledgebase:
                        knowledgebase[(w,1)] += 1
                    else:
                        knowledgebase[(w,1)] = 1

                    if not ((w,0) in knowledgebase):
                        knowledgebase[(w,0)] = 1
                    emailWord.append(w)
        rFile.close()


    # Bayes
    spamn = nspam +1.0
    hamn = nham +1.0
    ph = hamn/(spamn+hamn)
    ps = spamn /(spamn+hamn)
    knwb = dict()
    for key in knowledgebase:
        # ham =0,spam=1
        if (key[1] == 0):
            knwb[key] = knowledgebase[key] /(knowledgebase[key]+knowledgebase[key[0],1])

        else:
            knwb[key] =knowledgebase[key] /(knowledgebase[key]+knowledgebase[key[0],0])


    return knwb

def testData(file, knwb):
    # getting words in test email
    i =0
    for key in knwb:
        print key[0]+" ham:" +str(knwb[(key[0],0)])+" spam:"+str(knwb[(key[0],1)])
        i +=1
        if(i > 30):
            break;
    rFile = open(file,'r' )
    emailWord = []
    for line in rFile:
        nline = re.sub(r'[^A-Za-z ]',"",line)
        words = nline.split(' ')
        for w in words:
            if w == "":
                continue
            if not(w in emailWord):
                emailWord.append(w)
    rFile.close()

    ns = 0
    nh = 0

    for w in emailWord:
        if (w,0) in knwb:
            phamw = knwb[w,0]
            pspamw = knwb[w,1]
            ns += math.log(1 - pspamw) - math.log(pspamw)
            nh += math.log(1 - phamw) - math.log(phamw)
    print ns
    print nh
    pham = 1 /(1+ math.exp(nh))
    pspam = 1 /(1+ math.exp(ns))
    print pham
    print pspam
    if (pspam != 0):
        if (pham/pspam >= 1):
            print "ham\n"
        elif (pham/pspam < 1):
            print "spam\n"
    else:
        print "DUCK!"
main()
