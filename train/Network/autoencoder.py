# import torch
# import torch.nn.functional as F
import torch.nn as nn


class Conv1dEncoder2(nn.Module):

    def __init__(self, input_feature, latent_dim):
        super().__init__()
        self.relu2 = nn.ReLU()
        self.encoder = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(input_feature, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(),

            # nn.Linear(input_feature//4, input_feature//16),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class Decoder2(nn.Module):
    def __init__(self,latent_dim,output_feature):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(),

            # nn.Linear(input_feature//16, input_feature//4),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(latent_dim, output_feature),
            # nn.BatchNorm1d(output_feature),
            nn.ReLU()
        )

    def forward(self,x):
        return self.decoder(x)

class AutoEncoder2En1De(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim):
        super().__init__()
        output_dim1, output_dim2 = input_dim1, input_dim2
        self.encoder1 = Conv1dEncoder2(input_dim1, hidden_dim)
        self.encoder2 = Conv1dEncoder2(input_dim2, hidden_dim)
        self.decoder2 = Decoder2(hidden_dim, output_dim2)

    def translate_1_to_2(self, x):
        e1 = self.encoder1(x)
        d2 = self.decoder2(e1)
        return d2

    def get_embedding(self, x):
        h = self.encoder1(x)
        return h

    def forward(self, x):
        input_modality1 = x[0]
        input_modality2 = x[1]
        latent_embed_modality1 = self.encoder1(input_modality1)
        latent_embed_modality2 = self.encoder2(input_modality2)
        output_modality2_transform = self.decoder2(latent_embed_modality1)
        output_modality2_itself = self.decoder2(latent_embed_modality2)
        return output_modality2_transform, output_modality2_itself, latent_embed_modality1 - latent_embed_modality2
