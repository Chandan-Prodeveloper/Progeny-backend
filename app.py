# Suppress TensorFlow warnings - MUST be at the very top before any imports
import os
import warnings
import sys

# Suppress TensorFlow CPU and oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress absl warnings
warnings.filterwarnings('ignore', category=UserWarning, module='absl')

# Download models from Google Drive before starting the app
print("\n" + "="*70)
print("🌱 PROGENY BACKEND - Plant Disease Detection System")
print("="*70 + "\n")

try:
    from download_models import download_all_models
    download_all_models()
    print("✅ Model files downloaded. Will load on demand to save memory...\n")
except Exception as e:
    print(f"\n{'='*70}")
    print(f"❌ CRITICAL ERROR: Failed to download models")
    print(f"{'='*70}")
    print(f"\n{e}\n")
    sys.exit(1)

# Now import other libraries
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import gc

app = Flask(__name__)
CORS(app)

# Get the path to models directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

print(f"📁 Models directory: {MODELS_DIR}")
print(f"💾 Memory optimization: Models will be loaded on-demand\n")

# Model cache - stores only the currently loaded model
CURRENT_MODEL = {'crop': None, 'model': None, 'classes': None}
crop_types = ['apple', 'corn', 'potato', 'tomato']

CLASS_MAPPINGS = {
    'apple': ['Scab', 'Black Rot', 'Cedar Rust', 'Healthy'],
    'potato': ['Early Blight', 'Late Blight', 'Healthy'],
    'corn': ['Blight', 'Common Rust', 'Healthy'],
    'tomato': ['Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold', 'Target Spot', 'Healthy']
}

def load_model_on_demand(crop_type):
    """Load model only when needed and unload previous model"""
    global CURRENT_MODEL
    
    # If this model is already loaded, return it
    if CURRENT_MODEL['crop'] == crop_type and CURRENT_MODEL['model'] is not None:
        print(f"✓ Using cached {crop_type} model")
        return CURRENT_MODEL['model'], CURRENT_MODEL['classes']
    
    # Unload previous model to free memory
    if CURRENT_MODEL['model'] is not None:
        print(f"🗑️  Unloading {CURRENT_MODEL['crop']} model to free memory...")
        del CURRENT_MODEL['model']
        CURRENT_MODEL['model'] = None
        CURRENT_MODEL['crop'] = None
        CURRENT_MODEL['classes'] = None
        
        # Force garbage collection
        gc.collect()
        tf.keras.backend.clear_session()
    
    # Load new model
    try:
        model_path = os.path.join(MODELS_DIR, f'{crop_type}_model.h5')
        
        if not os.path.exists(model_path):
            raise Exception(f"Model file not found: {model_path}")
        
        print(f"📥 Loading {crop_type} model into memory...")
        model = tf.keras.models.load_model(model_path)
        classes = CLASS_MAPPINGS[crop_type]
        
        # Cache the loaded model
        CURRENT_MODEL['crop'] = crop_type
        CURRENT_MODEL['model'] = model
        CURRENT_MODEL['classes'] = classes
        
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ Loaded {crop_type} model ({file_size:.1f} MB, {len(classes)} classes)")
        
        return model, classes
        
    except Exception as e:
        print(f"❌ Error loading {crop_type} model: {e}")
        raise

print(f"\n{'='*70}")
print(f"🚀 Application Ready! Memory-optimized mode enabled")
print(f"   Available crops: {', '.join(crop_types)}")
print(f"{'='*70}\n")

# Disease remedies
DISEASE_REMEDIES = {
    'Healthy': [
        'Continue regular monitoring',
        'Maintain proper watering schedule',
        'Keep area clean and free of debris',
        'Ensure adequate spacing between plants'
    ],
    'Scab': [
        'Apply fungicides during wet weather',
        'Remove fallen leaves and infected fruit',
        'Prune trees to improve air circulation',
        'Choose resistant varieties when replanting'
    ],
    'Black Rot': [
        'Remove and destroy infected leaves and fruit',
        'Apply copper-based fungicides',
        'Improve air circulation around plants',
        'Avoid overhead watering'
    ],
    'Cedar Rust': [
        'Remove nearby cedar trees if possible',
        'Apply fungicides in early spring',
        'Plant resistant apple varieties',
        'Rake and destroy fallen leaves'
    ],
    'Blight': [
        'Apply appropriate fungicides',
        'Remove and destroy infected plant material',
        'Practice crop rotation',
        'Ensure proper spacing for air circulation'
    ],
    'Common Rust': [
        'Apply fungicides if infection is severe',
        'Plant resistant varieties',
        'Remove volunteer corn plants',
        'Monitor fields regularly'
    ],
    'Early Blight': [
        'Apply chlorothalonil or copper-based fungicides',
        'Remove lower leaves that touch the ground',
        'Mulch around plants to prevent soil splash',
        'Practice crop rotation',
        'Water at soil level, avoid wetting foliage'
    ],
    'Late Blight': [
        'Apply fungicides immediately upon detection',
        'Remove and destroy infected plants',
        'Avoid overhead irrigation',
        'Monitor weather conditions favorable to disease',
        'Ensure good air circulation'
    ],
    'Bacterial Spot': [
        'Apply copper-based bactericides',
        'Use disease-free seeds and transplants',
        'Avoid overhead watering',
        'Remove and destroy infected plants',
        'Practice crop rotation'
    ],
    'Leaf Mold': [
        'Improve ventilation in greenhouse or garden',
        'Reduce humidity levels',
        'Remove and destroy infected leaves',
        'Apply appropriate fungicides if needed',
        'Space plants properly for air flow'
    ],
    'Target Spot': [
        'Apply fungicides containing chlorothalonil',
        'Remove infected plant debris',
        'Improve air circulation',
        'Practice crop rotation',
        'Avoid working with plants when wet'
    ]
}

def read_file_as_image(data) -> np.ndarray:
    """Preprocess image for model input"""
    try:
        image = Image.open(io.BytesIO(data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((256, 256))
        image_array = np.array(image)
        
        return image_array
        
    except Exception as e:
        raise Exception(f"Error processing image: {e}")

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        'message': 'Progeny Backend API - Plant Disease Detection',
        'status': 'running',
        'memory_mode': 'optimized (on-demand loading)',
        'endpoints': {
            'health': '/health',
            'predict': '/predict (POST)'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    current = CURRENT_MODEL['crop'] if CURRENT_MODEL['crop'] else 'None'
    return jsonify({
        'status': 'healthy',
        'memory_mode': 'optimized',
        'currently_loaded_model': current,
        'available_crops': crop_types,
        'models_directory': MODELS_DIR
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint with on-demand model loading"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        crop_type = request.form.get('crop_type')
        
        print(f"\n{'='*60}")
        print(f"🌱 PREDICTION REQUEST")
        print(f"{'='*60}")
        print(f"Crop Type: {crop_type}")
        
        if not crop_type:
            return jsonify({'error': 'crop_type parameter is required'}), 400
        
        if crop_type not in crop_types:
            return jsonify({
                'error': f'Invalid crop type: {crop_type}',
                'available_crops': crop_types
            }), 400
        
        # Load model on demand
        model, class_names = load_model_on_demand(crop_type)
        
        # Read and preprocess image
        image_file = request.files['image']
        print(f"Image: {image_file.filename}")
        
        image = read_file_as_image(image_file.read())
        img_batch = np.expand_dims(image, 0)
        
        print(f"Image shape: {img_batch.shape}")
        
        # Get predictions
        predictions = model.predict(img_batch, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_idx]
        confidence = float(np.max(predictions[0]))
        
        # Log predictions
        print(f"\n🎯 PREDICTION RESULTS:")
        print(f"{'-'*60}")
        for idx, class_name in enumerate(class_names):
            conf = predictions[0][idx]
            bar = '█' * int(conf * 40)
            print(f"   {class_name:20s} → {conf:.4f} ({conf*100:5.1f}%) {bar}")
        
        print(f"{'-'*60}")
        print(f"🏆 FINAL: {predicted_class} ({confidence*100:.1f}%)")
        print(f"{'='*60}\n")
        
        # Create all predictions array
        all_predictions = [
            {
                'class': class_names[i], 
                'confidence': float(predictions[0][i])
            }
            for i in range(len(class_names))
        ]
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Get remedies
        remedies = DISEASE_REMEDIES.get(predicted_class, [
            'Consult with agricultural specialist',
            'Remove infected plant parts',
            'Monitor plants regularly',
            'Maintain proper watering and nutrition'
        ])
        
        return jsonify({
            'success': True,
            'crop_type': crop_type,
            'disease_name': predicted_class,
            'confidence_score': confidence,
            'remedies': remedies,
            'all_predictions': all_predictions
        })
        
    except Exception as e:
        print(f'\n❌ PREDICTION ERROR:')
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting server on port {port}...\n")
    app.run(host='0.0.0.0', port=port, debug=False)