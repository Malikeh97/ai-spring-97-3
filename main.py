from os import listdir
from Network import Network

ROOT = "notMNIST_large"
DS_STORE = ".DS_Store"


def createDataSet():
    folders = listdir(ROOT)
    if DS_STORE in folders:
        folders.remove(DS_STORE)

    dataSet = []
    for folder in folders:
        files = listdir("%s/%s" % (ROOT, folder))
        if DS_STORE in files:
            files.remove(DS_STORE)

        dataSet += [("%s/%s/%s" % (ROOT, folder, file)) for file in files]

    return dataSet


if __name__ == "__main__":
    ds = createDataSet()
    net = Network(ds, 30)