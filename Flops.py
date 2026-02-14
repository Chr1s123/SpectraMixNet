import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table

from models import SpectraMixNet_t

model = SpectraMixNet_t().cuda()
model.eval()

input = torch.randn(1, 3, 256, 256).cuda()

flops = FlopCountAnalysis(model, input)
print("Total FLOPs:", flops.total())
print(parameter_count_table(model))
