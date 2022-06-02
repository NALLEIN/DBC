file = "/home/jianghao/Code/Graduation/4k1/test/BD-Rate-260000-266-22-37.log"
if __name__ == '__main__':
    f = open(file, mode='r')

    lines = f.readlines()
    bdrate = []
    for i in range(12, len(lines), 13):
        # print(lines[i].rsplit(':  ')[-1])
        bdrate.append(float(lines[i].rsplit(':  ')[-1]))
    goodSum = 0
    goodCount = 0
    badSum = 0
    badCount = 0
    allSum = 0
    allCount = 0
    for i in range(len(bdrate)):
        if (abs(bdrate[i]) < 50.0):
            if bdrate[i] < 0:
                goodSum += bdrate[i]
                goodCount += 1
            else:
                badSum += bdrate[i]
                badCount += 1
            allSum += bdrate[i]
            allCount += 1
    print("good count %d, good avg %.3f ,  bad count %d, bad avg %.3f" % (goodCount, goodSum / goodCount, badCount, badSum))

    print("all count %d, all avg %.3f" % (allCount, allSum / allCount))