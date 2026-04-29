import os
import shutil
from pathlib import Path
from datetime import datetime
import uuid

# Create uploads directory
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

def save_upload_file(upload_file):
    """
    Save uploaded file to disk with unique filename.
    
    Args:
        upload_file: FastAPI UploadFile object
        
    Returns:
        Path to saved file
    """
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    file_extension = Path(upload_file.filename).suffix
    
    filename = f"mosquito_{timestamp}_{unique_id}{file_extension}"
    file_path = UPLOAD_DIR / filename
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    return file_path

def cleanup_upload_file(file_path):
    """
    Delete uploaded file after processing.
    
    Args:
        file_path: Path to file to delete
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up: {file_path}")
    except Exception as e:
        print(f"Warning: Could not delete {file_path}: {e}")

def validate_image_file(filename):
    """
    Validate if uploaded file is an image.
    
    Args:
        filename: Name of uploaded file
        
    Returns:
        Boolean indicating if file is valid image
    """
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    file_extension = Path(filename).suffix.lower()
    return file_extension in allowed_extensions