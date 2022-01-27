import torch

from models.superglue import SuperGlue

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Matching(torch.nn.Module):
    def __init__(self, config={}):
        super().__init__()
        self.superglue = SuperGlue()

    def forward(self, data_0, data_1):
        pred = {}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])


        # Perform the matching
        desc0 = torch.tensor(data_0['descriptors']).permute(1, 2, 0).to(device).float()
        kpts0 = torch.tensor(data_0['keypoints']).unsqueeze(0).to(device)
        scores0 = torch.tensor(data_0['scores']).unsqueeze(0).to(device)
        desc1 = torch.tensor(data_1['descriptors']).permute(1, 2, 0).to(device).float()
        kpts1 = torch.tensor(data_1['keypoints']).unsqueeze(0).to(device)
        scores1 = torch.tensor(data_1['scores']).unsqueeze(0).to(device)
        pred = {**pred, **self.superglue(data)}
        return pred
