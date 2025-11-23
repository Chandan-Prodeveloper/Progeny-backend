import os
import gdown
from pathlib import Path

# ⚠️ REPLACE THESE WITH YOUR ACTUAL GOOGLE DRIVE FILE IDs
# To get File ID from Google Drive link:
# https://drive.google.com/file/d/1a2B3c4D5e6F7g8H9i0J/view?usp=sharing
#                                 ^^^^^^^^^^^^^^^^^^^
#                                 This is your FILE_ID

MODEL_CONFIGS = {
   'apple_model.h5': '1njY0_1O1LkU_3RIJd-99HSSD_dpTv-g2',
    'corn_model.h5': '1iNLjeWiMCqjsbhVgETqbdR6-CCL4CGlP',
    'potato_model.h5': '1bnQoecm3U516e7iyZ9yQqzZdn4S7Jp9T',
    'tomato_model.h5': '1hMXKLJm_gis4oJpDrYiAPk7RaBXKtbhb',
}

def download_from_google_drive(file_id, destination):
    """Download a file from Google Drive"""
    if os.path.exists(destination):
        file_size = os.path.getsize(destination) / (1024 * 1024)  # MB
        print(f"✓ Model already exists: {os.path.basename(destination)} ({file_size:.1f} MB)")
        return True
    
    print(f"📥 Downloading {os.path.basename(destination)} from Google Drive...")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Download from Google Drive using gdown
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, destination, quiet=False)
        
        # Verify download
        if not os.path.exists(destination):
            raise Exception(f"File was not downloaded successfully")
        
        file_size = os.path.getsize(destination) / (1024 * 1024)  # MB
        print(f"✅ Successfully downloaded: {os.path.basename(destination)} ({file_size:.1f} MB)")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {os.path.basename(destination)}: {e}")
        print(f"   Make sure the file is shared publicly on Google Drive")
        print(f"   File ID: {file_id}")
        return False

def download_all_models():
    """Download all required models from Google Drive"""
    print("\n" + "="*70)
    print("🚀 Initializing ML Models from Google Drive...")
    print("="*70 + "\n")
    
    # Get the models directory path
    base_dir = Path(__file__).parent
    models_dir = base_dir / 'models'
    
    # Ensure models directory exists
    models_dir.mkdir(exist_ok=True)
    
    success_count = 0
    total_count = len(MODEL_CONFIGS)
    failed_models = []
    
    for filename, file_id in MODEL_CONFIGS.items():
        filepath = models_dir / filename
        
        # Check if FILE_ID is configured
        if file_id.startswith('YOUR_'):
            print(f"⚠️  {filename} - FILE_ID NOT CONFIGURED!")
            print(f"   Please update MODEL_CONFIGS in download_models.py")
            print(f"   Current value: {file_id}\n")
            failed_models.append(filename)
            continue
        
        if download_from_google_drive(file_id, str(filepath)):
            success_count += 1
        else:
            failed_models.append(filename)
        
        print()  # Empty line for readability
    
    print("="*70)
    print(f"📊 Download Summary: {success_count}/{total_count} models ready")
    
    if failed_models:
        print(f"❌ Failed models: {', '.join(failed_models)}")
    
    print("="*70 + "\n")
    
    if success_count < total_count:
        missing = total_count - success_count
        error_msg = f"Failed to download {missing} model(s): {', '.join(failed_models)}"
        error_msg += "\n\nPlease check:"
        error_msg += "\n1. File IDs are correctly configured in download_models.py"
        error_msg += "\n2. Files are shared publicly on Google Drive"
        error_msg += "\n3. Internet connection is stable"
        raise Exception(error_msg)
    
    print("✅ All models downloaded and ready!")
    return True

if __name__ == '__main__':
    try:
        download_all_models()
    except Exception as e:
        print(f"\n❌ Error: {e}\n")

        exit(1)
