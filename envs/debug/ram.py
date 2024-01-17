import psutil

# 获取内存信息
memory = psutil.virtual_memory()

# 可用内存
available_memory_gb = memory.available / (1024**3)
print(f'可用内存: {available_memory_gb:.2f} GB')
