import torch
import torch.nn as nn
import torch.optim as optim
from dataset import dataset, collate_fn
from model import Model
import numpy as np
import pandas as pd

def to_device(data, device):
    for k in data:
        data[k] = data[k].to(device)
    

nepoch = 10
device = "cuda:0"

train_data = dataset("human_cell.csv", "train")
training_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=1, 
        shuffle=True, 
        collate_fn=collate_fn
        )


test_data = dataset("human_cell.csv", "test")
test_loader = torch.utils.data.DataLoader(
        test_data, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=collate_fn
        )

model = Model().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_his = []
for epoch in range(nepoch):
    i = 0
    loss_train = []
    for d in training_loader:
        to_device(d, device)
        out = model(d)
        tmpred = out["tm"]
        loss = loss_fn(tmpred, d["tm"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1
        if i % 100 == 0:
            print (loss, "step", i)
        loss_train.append(loss.detach().cpu().numpy())
    print ("train", np.mean(loss_train), "epoch", epoch)
    loss_test = []
    with torch.no_grad():
        for td in test_loader:
            to_device(td, device)
            tmpred = model(td)["tm"]
            loss = loss_fn(tmpred, td["tm"])
            loss_test.append(loss.detach().cpu().numpy())
    print ("test", np.mean(loss_test), "epoch", epoch)
    loss_his.append([epoch, np.mean(loss_train), np.mean(loss_test)])

df = pd.DataFrame(loss_his)
df.columns = ["epoch", "loss_train", "loss_test"]
df.to_csv("loss_his.csv", index=False)
