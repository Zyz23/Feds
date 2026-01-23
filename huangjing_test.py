import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
# 输出应包含 "4090" 且版本号 >= 2.0