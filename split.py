import readline
from tracemalloc import start
import pandas as pd
import csv
from itertools import islice


def getAmountOfLineLenghts():
    num_lines = sum(1 for line in open("data/aventics_reallyreallynow.csv"))
    # result = 133925093
    return num_lines


def splitFile():
    start = 0
    end = 2678501
    counter = 0
    # do i need to read the whole file for removing the first lines of an file? python
    while True:
        counter += 1
        # open random csv
        with open("data/aventics_reallyreallynow.csv", "r") as myfile:
            # safe all rows from start to end
            head = list(islice(myfile, start, end))
            # get length off all this rows
            length = len(head)
        # if length == 0 break
        if length == 0:
            break
            # set start to end
        start = end
        end = end + 2678501
        # open new filee and write rows to it
        with open("output/new" + str(counter) + ".csv", "a") as f2:
            for item in head:
                f2.write(item)


def createFiles():
    for i in range(7, 11):
        with open("testFolder/" + "AventicsMonth" + str(i) + ".csv", "w") as file:
            with open("data/aventics_reallyreallynow.csv", "r") as file2:
                for j in range(4):
                    file.write(file2.readline())


def arrangeRows():
    # so in the case I have 60 csv files
    # for in range (1-51)
    # open csv file
    # if first file remove first four lines
    for i in range(1, 52):
        # can print number of line and name of current file for debugging
        with open("output/new" + str(i) + ".csv", "r") as file:
            lines = file.readlines()
        if i == 1:
            for i in range(4):
                lines.pop(0)
        for j in lines:
            try:
                month = j[51:53]
                if month[0] == str(0):
                    month = month[1]
                with open(
                    "testFolder/" + "AventicsMonth" + month + ".csv", "a"
                ) as file:
                    file.write(j)
            except IndexError:
                pass

def test():
    s = ",,1,2022-08-01T00:00:00Z,2022-10-19T00:00:00Z,2022-10"
    print(len(s))


if __name__ == "__main__":
    createFiles()
    arrangeRows()
    # splitFile()
    # print(getAmountOfLineLenghts())
    # test()
