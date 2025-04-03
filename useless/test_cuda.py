import torch
def test_cuda():
   torch.cuda.empty_cache()
   print(torch.cuda.nccl.version())
   print(torch.version.cuda) 
   print(f'PyTorch version: {torch.__version__}')
    # 检查 CUDA 是否可用
   print(f'CUDA is available: {torch.cuda.is_available()}')
   # 检查当前使用的设备
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   print(f'Using device: {device}')
   # 检查当前 GPU 的名称
   if torch.cuda.is_available():
      print(f'GPU Name: {torch.cuda.get_device_name(0)}')
   else:
      print('No GPU available.')