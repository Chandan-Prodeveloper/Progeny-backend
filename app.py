# Suppress TensorFlow warnings - MUST be at the very top before any imports
import os
import warnings
import sys

# Suppress TensorFlow CPU and oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

# Suppress absl warnings about compiled metrics
warnings.filterwarnings('ignore', category=UserWarning, module='absl')

# Download models from Google Drive before starting the app
print("\n" + "="*70)
print("🌱 PROGENY BACKEND - Plant Disease Detection System")
print("="*70 + "\n")

try:
    from download_models import download_all_models
    download_all_models()
    print("✅ Model initialization complete. Starting Flask application...\n")
except Exception as e:
    print(f"\n{'='*70}")
    print(f"❌ CRITICAL ERROR: Failed to initialize models")
    print(f"{'='*70}")
    print(f"\n{e}\n")
    print("Please check:")
    print("1. download_models.py has correct Google Drive File IDs")
    print("2. Models are shared publicly on Google Drive")
    print("3. gdown package is installed (check requirements.txt)")
    print(f"\n{'='*70}\n")
    sys.exit(1)

# Now import other libraries
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Get the path to models directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

print(f"📁 Models directory: {MODELS_DIR}\n")

# Load models at startup with class mappings
MODELS = {}
crop_types = ['apple', 'corn', 'potato', 'tomato']

CLASS_MAPPINGS = {
    'apple': ['Scab', 'Black Rot', 'Cedar Rust', 'Healthy'],
    'potato': ['Early Blight', 'Late Blight', 'Healthy'],
    'corn': ['Blight', 'Common Rust', 'Healthy'],
    'tomato': ['Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold', 'Target Spot', 'Healthy']
}

print("🔄 Loading TensorFlow models into memory...\n")

for crop in crop_types:
    try:
        model_path = os.path.join(MODELS_DIR, f'{crop}_model.h5')
        
        if not os.path.exists(model_path):
            print(f'❌ Model file not found: {model_path}')
            continue
        
        MODELS[crop] = {
            'model': tf.keras.models.load_model(model_path),
            'classes': CLASS_MAPPINGS[crop]
        }
        
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f'✅ Loaded {crop.capitalize()} model ({len(CLASS_MAPPINGS[crop])} classes, {file_size:.1f} MB)')
        
    except Exception as e:
        print(f'❌ Error loading {crop} model: {e}')

print(f"\n{'='*70}")
print(f"🚀 Application Ready! {len(MODELS)}/{len(crop_types)} models loaded")
print(f"{'='*70}\n")

if len(MODELS) == 0:
    print("❌ WARNING: No models loaded! Application will not work correctly.")
    print("Please check if models were downloaded successfully.\n")

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
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize((256, 256))
        
        # Convert to numpy array
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
        'endpoints': {
            'health': '/health',
            'predict': '/predict (POST)'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'models_loaded': list(MODELS.keys()),
        'total_models': len(MODELS),
        'models_directory': MODELS_DIR,
        'available_crops': crop_types
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Validate image file
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        # Validate crop type
        crop_type = request.form.get('crop_type')
        
        print(f"\n{'='*60}")
        print(f"🌱 PREDICTION REQUEST")
        print(f"{'='*60}")
        print(f"Crop Type: {crop_type}")
        
        if not crop_type:
            return jsonify({'error': 'crop_type parameter is required'}), 400
        
        if crop_type not in MODELS:
            return jsonify({
                'error': f'Invalid crop type: {crop_type}',
                'available_crops': list(MODELS.keys())
            }), 400
        
        # Get model and class names
        model_info = MODELS[crop_type]
        model = model_info['model']
        class_names = model_info['classes']
        
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
    # Get port from environment variable (for Render deployment)
    port = int(os.environ.get('PORT', 5000))
    
    # Check if models are loaded
    if len(MODELS) == 0:
        print("\n⚠️  WARNING: Starting server with no models loaded!")
        print("The API will not function correctly.\n")
    
    # Run the app
    print(f"🚀 Starting server on port {port}...\n")
    app.run(host='0.0.0.0', port=port, debug=False)