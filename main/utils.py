from PIL import Image
import base64
import io
import numpy as np

def numpy_to_base64(image_data):
    """Converts a numpy array image to a base64 encoded string."""
    # image_data is expected to be a 28x28 numpy array with values in [0, 1]
    image_data_uint8 = (np.clip(image_data, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(image_data_uint8, 'L') # L is for grayscale
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}" 