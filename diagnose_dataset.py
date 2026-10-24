import os
import pickle

def diagnose_dataset():
    # Your paths
    dataset_base = '/Users/marcellacung/Downloads/hdl_project/dataset'
    feature_dir = '/Users/marcellacung/Downloads/hdl_project/features'
    
    print("="*60)
    print("DATASET DIAGNOSIS")
    print("="*60)
    
    # Check if directories exist
    print(f"\n📁 Dataset base: {dataset_base}")
    print(f"   Exists: {os.path.exists(dataset_base)}")
    
    print(f"\n📁 Feature directory: {feature_dir}")
    print(f"   Exists: {os.path.exists(feature_dir)}")
    
    # List dataset contents
    if os.path.exists(dataset_base):
        print(f"\n📂 Contents of dataset directory:")
        for item in os.listdir(dataset_base):
            item_path = os.path.join(dataset_base, item)
            if os.path.isdir(item_path):
                print(f"   📁 {item}/")
                # List subdirectories or files
                subitems = os.listdir(item_path)
                for subitem in subitems[:10]:  # Show first 10
                    subitem_path = os.path.join(item_path, subitem)
                    if os.path.isdir(subitem_path):
                        print(f"      📁 {subitem}/")
                        # Show first 5 files in subdirectory
                        files = [f for f in os.listdir(subitem_path) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        if files:
                            print(f"        Sample images: {files[:3]}")
                    else:
                        if subitem.lower().endswith(('.jpg', '.jpeg', '.png')):
                            print(f"      🖼️  {subitem}")
                if len(subitems) > 10:
                    print(f"      ... and {len(subitems) - 10} more items")
            else:
                print(f"   📄 {item}")
    
    # Check feature files
    if os.path.exists(feature_dir):
        print(f"\n📊 Feature files:")
        pkl_files = [f for f in os.listdir(feature_dir) if f.endswith('.pkl')]
        for file in pkl_files:
            file_path = os.path.join(feature_dir, file)
            size = os.path.getsize(file_path)
            print(f"   📄 {file} ({size:,} bytes)")
            
            # Try to load and examine
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                print(f"      Keys: {list(data.keys())}")
                if 'name' in data:
                    print(f"      Sample names (first 5):")
                    for i, name in enumerate(data['name'][:5]):
                        print(f"        {i+1}. {name}")
                if 'feature' in data:
                    print(f"      Features shape: {data['feature'].shape}")
                
            except Exception as e:
                print(f"      Error loading: {e}")
    
    # Count total images
    print(f"\n🔢 Counting images in dataset...")
    total_images = 0
    image_extensions = ('.jpg', '.jpeg', '.png')
    
    if os.path.exists(dataset_base):
        for root, dirs, files in os.walk(dataset_base):
            image_files = [f for f in files if f.lower().endswith(image_extensions)]
            if image_files:
                print(f"   📁 {root}: {len(image_files)} images")
                total_images += len(image_files)
    
    print(f"\n✅ Total images found: {total_images}")
    print("="*60)

if __name__ == '__main__':
    diagnose_dataset()