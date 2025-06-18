import torch
import torch.nn as nn


class UncertaintyWeight(nn.Module):
    def __init__(self, task_num):
        super(UncertaintyWeight, self).__init__()
        self.task_num = task_num
        # self.ini_tensor = torch.tensor((-1.6, 0.)) #init with particular params
        # self.log_vars = nn.Parameter(self.ini_tensor) #init with particular params
        self.log_vars = nn.Parameter(
            torch.zeros(task_num)
        )  # init with zero，compute log sigma**2

    def forward(self, loss0, loss1):
        precision1 = torch.exp(-self.log_vars[0])
        loss = torch.sum(precision1 * loss0 + self.log_vars[0], -1)

        precision2 = torch.exp(-self.log_vars[1])
        loss += torch.sum(precision2 * loss1 + self.log_vars[1], -1)

        loss = torch.mean(loss)
        return loss, self.log_vars.data.tolist()
