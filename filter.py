import re
# 0 -> ham
# 1 -> spam
def main():
    spam = 1
    if spam:
        print "spam"
    else:
        print "ham"


def onlyWords(filename):
    rFile = open(filename[0],r )
    outFile = open(filename[1],w )
    for line in rFile:
        nline = re.sub(r'[^A-Za-z\s]',"",line)
        outFile.write(nline)

def buildTrainingSet():
    knowledgebase = dict()
    
    for i in range(400):
        filename = "ham"+("%03d"%x)+".txt"
        rFile = open(filename,r )
        for line in rFile:
            nline = re.sub(r'[^A-Za-z\s]',"",line)
            words = nline.split(' ')
            for w in words:
                if (w,0) in knowledgebase:
                    knowledgebase[(w,0)] += 1
                else:
                    knowledgebase[(w,0)] = 1
                    knowledgebase[(w,1)] = 0
        rFile.close()

    for i in range(100):
        filename = "spam"+("%03d"%x)+".txt"
        rFile = open(filename,r )
        for line in rFile:
            nline = re.sub(r'[^A-Za-z\s]',"",line)
            words = nline.split(' ')
            for w in words:
                if (w,1) in knowledgebase:
                    knowledgebase[(w,1)] += 1
                else:
                    knowledgebase[(w,1)] = 1

                if not ((w,0) in knowledgebase):
                    knowledgebase[(w,0)] = 0
        rFile.close()
