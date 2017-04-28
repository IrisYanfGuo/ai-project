
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

def

