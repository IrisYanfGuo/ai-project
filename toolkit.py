import time
import matplotlib.pyplot as plt

# toolkit for project including File I/O, graphs

def readDataSet(filename):
    # open dataset file to read data set
    attributes =[]
    instances = []
    print("start reading dataSet")
    try:
        f = open(filename)
        line = f.readline()
        while line:
            line = line.strip()
            if line.startswith("@attr") or line.startswith("@ATTR"):
                attributes.append(line.split()[1])
            if not line.startswith("@") and len(line) > 2:
                instances.append(line.strip().split(","))
            line = f.readline()
        f.close()

    except Exception as e:
        print("open file error")
    print("reading completed!")

    return attributes, instances

def draw_bar(accuracy_list,save = False):

    d= accuracy_list
    plt.bar(range(len(d)), d)

    for t in range(len(d)):
        ss = str(d[t])
        ss = ss[:4]
        plt.text(t, d[t], ss, ha='center', va='bottom')

    plt.tight_layout()  # disjust y label
    plt.xticks(range(len(d)), range(1, len(d) + 1))  # set x bar label

    if save:
        save_img()
    plt.show()


def save_img():
    name = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    name = name + ".png"
    plt.savefig(name)







