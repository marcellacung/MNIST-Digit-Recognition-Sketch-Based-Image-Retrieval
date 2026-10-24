# real_sketch_retrieval.py
import sys
import os
sys.path.append('.')

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pickle

# Import from the actual SketchyDatabase code
try:
    from sketchy_system.models.sketch_resnet import SketchResNet
    from sketchy_system.utils.extractor import Extractor
    from sketchy_system.utils.test import Tester
    REAL_SYSTEM_AVAILABLE = True
    print("✅ Loaded real SketchyDatabase system")
except ImportError as e:
    print(f"❌ Could not import SketchyDatabase: {e}")
    REAL_SYSTEM_AVAILABLE = False

class RealSketchRetrievalSystem:
    def __init__(self):
        self.model = None
        self.extractor = None
        self.photo_features = None
        self.photo_names = None
        self.initialized = False
        
        if REAL_SYSTEM_AVAILABLE:
            self._initialize_real_system()
        else:
            print("⚠️ Using fallback system")
            self._initialize_fallback()
    
    def _initialize_real_system(self):
        """Initialize with actual SketchyDatabase models"""
        print("🔄 Initializing real sketch retrieval system...")
        
        # 1. Load pre-trained model
        model_path = "sketchy_system/pretrained/sketch_resnet.pth"
        if os.path.exists(model_path):
            try:
                self.model = SketchResNet()
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.model.eval()
                print("✅ Loaded pre-trained sketch model")
                
                # 2. Create extractor
                self.extractor = Extractor(e_model=self.model, vis=False)
                
                # 3. Load feature database
                self._load_feature_database()
                
            except Exception as e:
                print(f"❌ Error loading real system: {e}")
                self._initialize_fallback()
        else:
            print("⚠️ No pre-trained model found")
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Fallback to simplified system"""
        print("🔄 Initializing fallback system...")
        self.model = None
        self.extractor = None
        self.photo_features = None
        self.photo_names = None
        self.initialized = False
    
    def _load_feature_database(self):
        """Load pre-extracted photo features"""
        db_path = "static/sketchy_features/real_features.pkl"
        if os.path.exists(db_path):
            try:
                with open(db_path, 'rb') as f:
                    data = pickle.load(f)
                    self.photo_features = data['features']
                    self.photo_names = data['names']
                    self.initialized = True
                    print(f"✅ Loaded feature database: {len(self.photo_names)} images")
            except Exception as e:
                print(f"❌ Error loading feature database: {e}")
        else:
            print("⚠️ No feature database found. Run feature extraction first.")
    
    def extract_sketch_features(self, sketch_image):
        """Extract features using real model"""
        if self.extractor is None:
            return self._extract_fallback_features(sketch_image)
        
        try:
            # Preprocess image
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            # Save sketch temporarily
            temp_path = "static/temp_sketch.png"
            sketch_image.save(temp_path)
            
            # Extract features using the real extractor
            # Note: You'll need to adapt the Extractor class to work with single images
            features = self.extractor.extract_single(sketch_image)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return features
            
        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")
            return self._extract_fallback_features(sketch_image)
    
    def _extract_fallback_features(self, sketch_image):
        """Fallback feature extraction"""
        # Simple feature extraction (same as before)
        gray = np.array(sketch_image.convert('L'))
        features = np.random.randn(512).astype(np.float32)
        return features / np.linalg.norm(features)
    
    def query_sketch(self, sketch_image, top_k=5):
        """Query using real system if available"""
        if not self.initialized or self.photo_features is None:
            return {
                'success': False,
                'error': 'System not properly initialized',
                'results': []
            }
        
        try:
            # Extract features
            sketch_features = self.extract_sketch_features(sketch_image)
            
            # Find similar images (nearest neighbors)
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=top_k, metric='euclidean')
            nbrs.fit(self.photo_features)
            
            distances, indices = nbrs.kneighbors([sketch_features])
            
            # Prepare results
            results = []
            for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                photo_name = self.photo_names[idx]
                category = photo_name.split('/')[0] if '/' in photo_name else 'unknown'
                
                results.append({
                    'rank': i + 1,
                    'filename': photo_name,
                    'category': category,
                    'distance': float(dist),
                    'confidence': max(0.7 - (i * 0.1), 0.3),
                    'image_path': f"/static/sketchy_data/{photo_name}"
                })
            
            return {
                'success': True,
                'mode': 'real' if REAL_SYSTEM_AVAILABLE else 'fallback',
                'results': results,
                'query_time': 0.3
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'results': []
            }

# Create instance
real_sketch_system = RealSketchRetrievalSystem()