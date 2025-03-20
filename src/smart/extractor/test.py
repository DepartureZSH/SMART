import torch
print(torch.__version__) # 查看torch版本
print(torch.cuda.is_available()) # 看安装好的torch和cuda能不能用，也就是看GPU能不能用

print(torch.version.cuda) # 输出一个 cuda 版本，注意：上述输出的 cuda 的版本并不一定是 Pytorch 在实际系统上运行时使用的 cuda 版本，而是编译该 Pytorch release 版本时使用的 cuda 版本，详见：https://blog.csdn.net/xiqi4145/article/details/110254093

import torch.utils
import torch.utils.cpp_extension
print(torch.utils.cpp_extension.CUDA_HOME) #输出 Pytorch 运行时使用的 cuda