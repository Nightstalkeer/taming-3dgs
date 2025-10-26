
print('Testing CUDA submodule imports...\n')

try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    print('✓ diff_gaussian_rasterization works')
except Exception as e:
    print(f'✗ diff_gaussian_rasterization FAILED: {e}')
    exit(1)

try:
    from simple_knn._C import distCUDA2

    print('✓ simple_knn works')
except Exception as e:
    print(f'✗ simple_knn FAILED: {e}')
    exit(1)

try:
    import fused_ssim

    print('✓ fused_ssim works')
except Exception as e:
    print(f'✗ fused_ssim FAILED: {e}')
    exit(1)

print('\n🎉 All CUDA submodules loaded successfully!')
print('\nEnvironment Summary:')
import torch

print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.version.cuda}')
print(f'  GPU: {torch.cuda.get_device_name(0)}')
print(f'  Compute Capability: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}')
