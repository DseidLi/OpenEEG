import torch
import time

# 检查是否有可用的 GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"有可用的 GPU：{torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("没有可用的 GPU，使用 CPU。")

# 创建一个随机张量并将其移动到选定的设备上
size = 1000
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

# 记录开始时间
start_time = time.time()

# 执行一些简单的运算
c = torch.matmul(a, b)

# 记录结束时间
end_time = time.time()

# 输出运算时间
print(f"运算耗时：{end_time - start_time} 秒")
