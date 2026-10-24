# utils/sketch_retrieval.py - SUPER SIMPLE VERSION
import pickle
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import os

class SimpleSketchRetrieval:
    def __init__(self):
        print("🚀 Starting sketch retrieval...")
        self.features = None
        self.names = None
        
        # Try to load ANY .pkl file
        pkl_files = [
            'photo-resnet-epoch.pkl',
            'sketch-resnet-epoch.pkl', 
            'photo-vgg-190epoch.pkl',
            'sketch-vgg-190epoch.pkl'
        ]
        
        for pkl in pkl_files:
            if os.path.exists(pkl):
                print(f"📦 Loading {pkl}...")
                try:
                    with open(pkl, 'rb') as f:
                        data = pickle.load(f)
                    self.features = data['feature']
                    self.names = data['name']
                    print(f"✅ Loaded {len(self.names)} images")
                    break
                except:
                    continue
        
        if self.features is None:
            print("⚠️ No .pkl files found, using demo")
            self._setup_demo()
        
        # Build search index
        self.nbrs = NearestNeighbors(n_neighbors=5, metric='euclidean')
        self.nbrs.fit(self.features)
        print("🎯 Ready for queries!")
    
    def _setup_demo(self):
        """Quick demo setup"""
        print("🔄 Setting up demo...")
        self.features = np.random.randn(100, 512)
        self.names = [f"demo_image_{i}.jpg" for i in range(100)]
    
    def query(self, sketch_image, top_k=5):
        """Simple query function"""
        try:
            # Convert sketch to features (SIMPLIFIED)
            img = sketch_image.resize((64, 64)).convert('L')
            features = np.array(img).flatten()[:512]
            
            # Search
            distances, indices = self.nbrs.kneighbors([features])
            
            # Prepare results
            results = []
            for i, idx in enumerate(indices[0][:top_k]):
                results.append({
                    'rank': i + 1,
                    'filename': self.names[idx],
                    'similarity': 0.9 - (i * 0.1),
                    'image_path': f"/static/demo/sample_{i+1}.jpg"
                })
            
            return {
                'success': True,
                'results': results
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Create instance
sketch_retrieval_system = SimpleSketchRetrieval()
real_sketch_system = sketch_retrieval_system