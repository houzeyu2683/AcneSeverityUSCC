
import data

train = "~/Desktop/Projects/Classification/AcneSeverity/resource/txt/train.txt"
test = "~/Desktop/Projects/Classification/AcneSeverity/resource/txt/test.txt"

table = data.table(train=train, test=test)

table.train[1] = table.train.loc[:,1].replace({"2":'1', "3":"1"})
table.test[1] = table.test.loc[:,1].replace({"2":'1', "3":"1"})
table.train.to_csv("tr.txt", sep='\t', index=False, header=None)
table.test.to_csv("te.txt", sep='\t', index=False, header=None)

def criterion(lesion):

    if lesion <= 5: return 0
    elif lesion <= 20: return 1
    elif lesion <= 50: return 2
    return 3

for l, i in zip(table.train[1], table.train[2]):

    j = criterion(int(i))
    print(l, j , int(l)==j)
    continue

import pandas
import os
import PIL.Image
import numpy 
import tqdm
loop = pandas.concat([table.train, table.test], axis=0).reset_index(drop=True)[0]
total = len(loop)
# r, g, b = [0,0], [0,0], [0,0], [0,0]
r_mean, g_mean, b_mean = 0, 0, 0
r_std, g_std, b_std = 0, 0, 0
for i in tqdm.tqdm(loop, total=total):
    # i = loop[0]
    source = './resource/jpg'
    # size = (224, 224)
    image = PIL.Image.open(os.path.join(source, i)).convert('RGB')#.resize(size)
    channel = numpy.array(image)[:,:,0], numpy.array(image)[:,:,1], numpy.array(image)[:,:,2]
    
    r_mean += numpy.array(image)[:,:,0].mean()
    g_mean += numpy.array(image)[:,:,1].mean()
    b_mean += numpy.array(image)[:,:,2].mean()
    r_std += numpy.array(image)[:,:,0].std()
    g_std += numpy.array(image)[:,:,1].std()
    b_std += numpy.array(image)[:,:,2].std()
    continue

r_mean = r_mean / total
g_mean = g_mean / total
b_mean = b_mean / total
r_std = r_std / total
g_std = g_std / total
b_std = b_std / total



