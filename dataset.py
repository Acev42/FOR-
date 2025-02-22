import numpy as np
import sys, os
import torch
import pandas as pd
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

class dataset(torch.utils.data.Dataset):
    def __init__(self, fcsv, tag = "train"):
        super(dataset,self).__init__()
        assert tag in ["train", "test", "vali"]
        #sequence,target,set,validation
        self.df = pd.read_csv(fcsv, sep=",")
        if tag == "test":
            self.df = self.df[self.df.set == tag]
        elif tag == "train":
            self.df = self.df[(self.df.set == tag) & (self.df.validation == False)][0:1500]
        elif tag == "validation":
            self.df = self.df[(self.df.set == "train") & (self.df.validation == True)]


        aaname = list('ACDEFGHIKLMNPQRSTVWYUXZO')
        esm_tokenid = [5,23,13,9,18,6,21,12,15,4,20,17,14,16,10,8,11,7,22,19,5,5,5,5]
        self.esm_map = dict(zip(aaname, esm_tokenid))

    def get_esmtoken(self, resname):
        esmtoken = list(map(lambda n: self.esm_map[n], resname))
        return np.array(esmtoken, dtype=int)
        #begin = 0
        #padding = 1
        #eos_token = 2
        #prepend = "<null_0>", "<pad>", "<eos>", "<unk>"
        #append_toks: Sequence[str] = ("<cls>", "<mask>", "<sep>")
   
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        data = {}
        data["esmtoken"] = self.get_esmtoken(self.df.sequence.iloc[index])
        if len(data["esmtoken"]) >= 1022:
            data["esmtoken"] = data["esmtoken"][0:1022]
        assert len(data["esmtoken"]) <= 1022
        data["esmtoken"] = np.pad(data["esmtoken"], (1, 0), constant_values=0) #pad 0 to left, bos token
        data["esmtoken"] = np.pad(data["esmtoken"], (0, 1), constant_values=2) #pad 2 to right, eos token
        data["esmtoken"] = np.expand_dims(data["esmtoken"], axis=0)
        data["tm"] = [self.df.target.iloc[index]]
        return data

def collate_fn(data):
    d = data[0]
    return {"esmtoken":torch.tensor(d["esmtoken"], dtype=torch.int64),
            "tm":torch.tensor(d["tm"], dtype=torch.float)}

if __name__ == "__main__":
    fcsv = sys.argv[1]
    data = dataset(fcsv, "test")
    print (len(data), "total samples")
    train_loader = DataLoader(data, batch_size=1, num_workers=4, shuffle=False, collate_fn=collate_fn)
    c = 0
    for d in train_loader:
        for k, v in d.items():
            print (k, v.shape, v.dtype, v)
        c+= 1
        if c>3:
            break
    #print (c)
