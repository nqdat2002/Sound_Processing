# find all of the files in the directory
import os

birds = []  # list of all birds (Ember, 3Phyll...)
blist = []  # list of files for one bird
flist = []  # list of all files for all birds. A list of blists.
for root, dirs, files in os.walk("./mels-5class/"):
    if root == "./mels-5class/":
        birds = dirs

for bird in birds:
    for root, dirs, files in os.walk("./mels-5class/" + bird):
        for file in files:
            if file.endswith(".png"):
                blist.append(os.path.join(root, file))
    print("Found ", len(blist), ' files for ', bird)
    flist.append(blist)
    blist = []

print("End!")
