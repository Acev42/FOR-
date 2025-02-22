from mlp import MultiLayerPerceptron
from torch import nn
import esm
import torch

class Model(nn.Module):
    def __init__(self, esmmodel_name = "esm2_t6_8M_UR50D"):
        super().__init__()
        
        esmmodel, alphabet =  esm.pretrained.load_model_and_alphabet(esmmodel_name)
        self.esmlayerid = esmmodel.num_layers
        self.esmdim = esmmodel.embed_dim

        self.esmmodel = esmmodel

        hidden_dims = [self.esmdim, 1]
        self.mlp = MultiLayerPerceptron(
                input_dim=self.esmdim, 
                hidden_dims=hidden_dims, 
                short_cut=False, 
                batch_norm=False, 
                activation="relu", 
                dropout=0)

    def forward(self, batch):
        esmtoken = batch["esmtoken"]
        assert esmtoken.shape[0] == 1 #batch size is 1
        with torch.no_grad():
            esmout = self.esmmodel(esmtoken, repr_layers=[self.esmlayerid], need_head_weights=False)
        esmrep = esmout['representations'][self.esmlayerid]
        esmrep_mean = torch.mean(esmrep[:, 1:-1, :], dim=1)
        tmpred = self.mlp(esmrep_mean)
        tmpred = torch.squeeze(tmpred, -1)
        return {"tm":tmpred}
 
