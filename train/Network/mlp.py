# import torch
import torch.nn as nn
class Net(nn.Module):
    def __init__(self, in_feature, **kwargs):
        super(Net, self).__init__()
        self.hid_feats = kwargs["hidden_size"]
        # self.nrc = kwargs["no_readout_concatenate"]
        # self.conv_layer_size = kwargs["conv_layers"]
        # self.relu = nn.GELU()
        # self.fc1 = nn.Linear(in_feature, self.hid_feats)
        # # self.dp = nn.Dropout(p=0.5)
        # self.bn1 = nn.GroupNorm(4, self.hid_feats)
        # self.fc2 = nn.Linear(self.hid_feats, self.hid_feats)
        # self.bn2 = nn.GroupNorm(4, self.hid_feats)
        # self.fc3 = nn.Linear(self.hid_feats, self.hid_feats)

        self.output = nn.Sequential(nn.Dropout(p=0.5),
                                nn.Linear(in_feature, self.hid_feats),
                                # nn.GroupNorm(4, self.hid_feats),
                                nn.BatchNorm1d(self.hid_feats),
                                # nn.Dropout(p=0.5),
                                nn.LeakyReLU(),
                                nn.Linear(self.hid_feats, self.hid_feats),
                                # nn.GroupNorm(4, self.hid_feats),
                                nn.BatchNorm1d(self.hid_feats),
                                # nn.Linear(self.hid_feats, self.hid_feats),
                                nn.LeakyReLU())

    def forward(self, x):
        # x = self.fc1(x)
        # x = self.bn1(x)
        #
        # x = self.relu(x)
        # x = self.fc2(x)
        # x = self.bn2(x)
        #
        # x = self.relu(x)
        # x = self.fc3(x)
        x = self.output(x)

        return x


class Net_res(nn.Module):
    def __init__(self, num_batches):
        super(Net_res, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(50, 500)
        self.bn1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 500)
        self.bn2 = nn.BatchNorm1d(500)
        self.fc3 = nn.Linear(500, 500)
        self.bn3 = nn.BatchNorm1d(500)
        self.fc4 = nn.Linear(500, 500)
        self.bn4 = nn.BatchNorm1d(500)
        self.fc5 = nn.Linear(500, 50)

        self.btch_classifier = nn.Linear(500, num_batches)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(self.bn1(x))

        x = x + self.fc2(x)
        x = self.relu(self.bn2(x))

        x = x + self.fc3(x)
        x = self.relu(self.bn3(x))

        btch = self.btch_classifier(x)

        x = x + self.fc4(x)
        x = self.relu(self.bn4(x))

        x = self.fc5(x)

        return x, btch