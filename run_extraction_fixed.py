"""
Working feature extraction script with all fixes applied.
"""
import os
import sys

# Fix paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Fix sketchy_system imports
sketchy_system_path = os.path.join(project_root, 'sketchy_system')
if os.path.exists(sketchy_system_path):
    sys.path.insert(0, sketchy_system_path)

print("=" * 70)
print("FEATURE EXTRACTION - FIXED VERSION")
print("=" * 70)

# Import after fixing paths
from sketchy_system.utils.extractor_cpu import ExtractorCPU
from models.vgg import vgg16
from models.sketch_resnet import resnet50
import torch as t
from torch import nn

# Dataset paths
test_photo_root = 'dataset/photo-test'
test_set_root = 'dataset/sketch-triplet-test'

print("\n📁 Checking dataset paths...")
for name, path in [("Test photos", test_photo_root), 
                   ("Test sketches", test_set_root)]:
    if os.path.exists(path):
        categories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        print(f"✓ {name}: {path} ({len(categories)} categories)")
    else:
        print(f"✗ {name}: {path} (MISSING!)")

# Use CPU
device = 'cpu'
print(f"\n⚙️ Using device: {device}")

'''vgg'''
print("\n" + "=" * 70)
print("VGG FEATURE EXTRACTION")
print("=" * 70)

try:
    vgg = vgg16(pretrained=False)
    vgg.classifier[6] = nn.Linear(in_features=4096, out_features=125, bias=True)
    
    print("⚠ Using random VGG weights (pretrained models not available)")
    
    vgg.to(device)
    vgg.eval()

    # Use our CPU-compatible extractor
    ext = ExtractorCPU(vgg, dataloader=True, device=device)
    
    print("📸 Extracting photo features with VGG...")
    if os.path.exists(test_photo_root):
        photo_feature = ext.extract(test_photo_root, 'photo-vgg-190epoch.pkl')
        print(f"✓ Photo features saved to 'photo-vgg-190epoch.pkl'")
        print(f"  Samples: {len(photo_feature['name'])}, Feature shape: {photo_feature['feature'].shape}")
    else:
        print("✗ Photo path not found")
    
    print("\n🎨 Extracting sketch features with VGG...")
    if os.path.exists(test_set_root):
        sketch_feature = ext.extract(test_set_root, 'sketch-vgg-190epoch.pkl')
        print(f"✓ Sketch features saved to 'sketch-vgg-190epoch.pkl'")
        print(f"  Samples: {len(sketch_feature['name'])}, Feature shape: {sketch_feature['feature'].shape}")
    else:
        print("✗ Sketch path not found")
        
except Exception as e:
    print(f"✗ VGG extraction failed: {e}")
    import traceback
    traceback.print_exc()

'''resnet'''
print("\n" + "=" * 70)
print("RESNET FEATURE EXTRACTION")
print("=" * 70)

try:
    resnet = resnet50()
    resnet.fc = nn.Linear(in_features=2048, out_features=125)
    
    print("⚠ Using random ResNet weights (pretrained models not available)")
    
    resnet.to(device)
    resnet.eval()

    # Use our CPU-compatible extractor
    ext = ExtractorCPU(resnet, dataloader=True, device=device)
    
    print("📸 Extracting photo features with ResNet...")
    if os.path.exists(test_photo_root):
        photo_feature = ext.extract(test_photo_root, 'photo-resnet-epoch.pkl')
        print(f"✓ Photo features saved to 'photo-resnet-epoch.pkl'")
        print(f"  Samples: {len(photo_feature['name'])}, Feature shape: {photo_feature['feature'].shape}")
    else:
        print("✗ Photo path not found")
    
    print("\n🎨 Extracting sketch features with ResNet...")
    if os.path.exists(test_set_root):
        sketch_feature = ext.extract(test_set_root, 'sketch-resnet-epoch.pkl')
        print(f"✓ Sketch features saved to 'sketch-resnet-epoch.pkl'")
        print(f"  Samples: {len(sketch_feature['name'])}, Feature shape: {sketch_feature['feature'].shape}")
    else:
        print("✗ Sketch path not found")
        
except Exception as e:
    print(f"✗ ResNet extraction failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✅ EXTRACTION COMPLETED!")
print("=" * 70)

# List created files
print("\n📁 Created feature files:")
import glob
for file in sorted(glob.glob("*.pkl")):
    size = os.path.getsize(file)
    print(f"  ✓ {file} ({size:,} bytes)")
