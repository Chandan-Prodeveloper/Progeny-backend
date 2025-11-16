# Plant Disease Detection Backend

Flask-based ML service for plant disease detection using TensorFlow.

## Setup

1. Create virtual environment:
```bash
python -m venv venv
```

2. Activate virtual environment:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Server

```bash
python app.py
```
Server will start on http://localhost:5000

### Health Check
```
GET /health
```

### Predict Disease
```
POST /predict
Content-Type: multipart/form-data

Parameters:
- image: Image file
- crop_type: One of ['apple', 'corn', 'potato', 'tomato']
```

## Environment Variables

Create a `.env` file based on `.env.example`:
- `FLASK_ENV`: development or production
- `PORT`: Server port (default: 5000)
- `HOST`: Server host (default: 0.0.0.0)
