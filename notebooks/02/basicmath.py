
def div(n1, n2):
    return n1 / n2



















def sortlist(lst):
    alist = lst[:]  # copy the list
    for passnum in range(len(alist)-1, 0, -1):
        for i in range(passnum):
            if alist[i] > alist[i+1]:
                temp = alist[i]
                alist[i] = alist[i+1]
                alist[i+1] = temp
    return alist









import os 
def openfile(filename): 
    # if not os.path.exists(filename): 
    #    return 

    open(filename) 

