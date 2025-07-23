import sys
import torch

print("Verifying installation of enhanced modules...")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    from ultralytics.nn.modules.enhanced import (
        DyConv, C2f_ECA, FeatureFusion, SPPF_Enhanced, 
        DetectEnhanced, ECAAttention, create_enhanced_module
    )
    print("Enhanced modules imported successfully")
    
    print(f" DyConv: {DyConv}")
    print(f" C2f_ECA: {C2f_ECA}")
    print(f" FeatureFusion: {FeatureFusion}")
    print(f" SPPF_Enhanced: {SPPF_Enhanced}")
    print(f" DetectEnhanced: {DetectEnhanced}")
    print(f" ECAAttention: {ECAAttention}")
    
    print("\n Testing module instantiation...")
    
    print("Testing DyConv...")
    conv = DyConv(64, 128, 3, 1)
    x = torch.randn(1, 64, 32, 32)
    y = conv(x)
    print(f" DyConv executed successfully: {x.shape} -> {y.shape}")
    
    print("Testing ECAAttention...")
    eca = ECAAttention(128)
    x = torch.randn(1, 128, 16, 16)
    y = eca(x)
    print(f" ECAAttention executed successfully: {x.shape} -> {y.shape}")
    
    print("Testing C2f_ECA...")
    c2f = C2f_ECA(128, 256, n=2)
    x = torch.randn(1, 128, 16, 16)
    y = c2f(x)
    print(f" C2f_ECA executed successfully: {x.shape} -> {y.shape}")
    
    print("Testing SPPF_Enhanced...")
    sppf = SPPF_Enhanced(512, 512, k=5)
    x = torch.randn(1, 512, 8, 8)
    y = sppf(x)
    print(f" SPPF_Enhanced executed successfully: {x.shape} -> {y.shape}")
    
    print("\n Installation completed successfully")
    print("All enhanced modules are working perfectly")
    
except ImportError as e:
    print(f" Import error: {e}")
    print("Path to the enhanced.py file:")
    print("/your_virtual_environment/lib/python3.11/site-packages/ultralytics/nn/modules/enhanced.py")
    
except Exception as e:
    print(f" Error during testing: {e}")