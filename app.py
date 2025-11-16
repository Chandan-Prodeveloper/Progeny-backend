# Suppress TensorFlow warnings - MUST be at the very top before any imports
import os
import warnings

# Suppress TensorFlow CPU and oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

# Suppress absl warnings about compiled metrics
warnings.filterwarnings('ignore', category=UserWarning, module='absl')

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

print(f"Looking for models in: {MODELS_DIR}")

# Load models at startup with class mappings
MODELS = {}
crop_types = ['apple', 'corn', 'potato', 'tomato']

CLASS_MAPPINGS = {
    'apple': ['Scab', 'Black Rot', 'Cedar Rust', 'Healthy'],
    'potato': ['Early Blight', 'Late Blight', 'Healthy'],
    'corn': ['Blight', 'Common Rust', 'Healthy'],
    'tomato': ['Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold', 'Target Spot', 'Healthy']
}

for crop in crop_types:
    try:
        model_path = os.path.join(MODELS_DIR, f'{crop}_model.h5')
        MODELS[crop] = {
            'model': tf.keras.models.load_model(model_path),
            'classes': CLASS_MAPPINGS[crop]
        }
        print(f'✓ Loaded {crop} model from {model_path}')
    except Exception as e:
        print(f'✗ Error loading {crop} model: {e}')

# Disease remedies
DISEASE_REMEDIES = {
    'Healthy': ['Continue regular monitoring', 'Maintain proper watering', 'Keep area clean'],
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
        'Practice crop rotation'
    ],
    'Late Blight': [
        'Apply fungicides immediately upon detection',
        'Remove and destroy infected plants',
        'Avoid overhead irrigation',
        'Monitor weather conditions favorable to disease'
    ],
    'Bacterial Spot': [
        'Apply copper-based bactericides',
        'Use disease-free seeds and transplants',
        'Avoid overhead watering',
        'Remove and destroy infected plants'
    ],
    'Leaf Mold': [
        'Improve ventilation in greenhouse or garden',
        'Reduce humidity levels',
        'Remove and destroy infected leaves',
        'Apply appropriate fungicides if needed'
    ],
    'Target Spot': [
        'Apply fungicides containing chlorothalonil',
        'Remove infected plant debris',
        'Improve air circulation',
        'Practice crop rotation'
    ]
}

def read_file_as_image(data) -> np.ndarray:
    """Preprocess image for model input"""
    image = Image.open(io.BytesIO(data))
    image = image.resize((256, 256))
    image_array = np.array(image)
    return image_array

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy', 
        'models_loaded': list(MODELS.keys()),
        'models_directory': MODELS_DIR
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        crop_type = request.form.get('crop_type')
        
        print(f"\n{'='*60}")
        print(f"🌱 CROP TYPE RECEIVED: {crop_type}")
        print(f"{'='*60}")
        
        if not crop_type or crop_type not in MODELS:
            return jsonify({'error': f'Invalid crop type. Must be one of: {list(MODELS.keys())}'}), 400
        
        model_info = MODELS[crop_type]
        model = model_info['model']
        class_names = model_info['classes']
        
        # Read and preprocess image
        image_file = request.files['image']
        image = read_file_as_image(image_file.read())
        img_batch = np.expand_dims(image, 0)
        
        print(f"📷 Image shape: {img_batch.shape}")
        
        # Get predictions
        predictions = model.predict(img_batch, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_idx]
        confidence = float(np.max(predictions[0]))
        
        print(f"\n🎯 PREDICTION PROBABILITIES:")
        print(f"{'-'*60}")
        for idx, class_name in enumerate(class_names):
            conf = predictions[0][idx]
            bar = '█' * int(conf * 50)
            print(f"   {idx}. {class_name:20s} → {conf:.4f} ({conf*100:5.1f}%) {bar}")
        
        print(f"{'-'*60}")
        print(f"🏆 TOP PREDICTION: {predicted_class} ({confidence*100:.1f}%)")
        print(f"{'='*60}\n")
        
        # Create all predictions array
        all_predictions = [
            {'class': class_names[i], 'confidence': float(predictions[0][i])}
            for i in range(len(class_names))
        ]
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Get remedies
        remedies = DISEASE_REMEDIES.get(predicted_class, [
            'Consult with agricultural specialist',
            'Remove infected plant parts',
            'Monitor plants regularly'
        ])
        
        return jsonify({
            'disease_name': predicted_class,
            'confidence_score': confidence,
            'remedies': remedies,
            'all_predictions': all_predictions
        })
        
    except Exception as e:
        print(f'\n❌ PREDICTION ERROR: {e}')
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
