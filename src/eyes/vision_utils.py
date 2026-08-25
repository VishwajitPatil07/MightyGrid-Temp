from PIL import Image

def resize_screenshot(input_path, output_path, max_width=512):
    """Utility function to compress screenshots on demand."""
    with Image.open(input_path) as img:
        w_percent = (max_width / float(img.size[0]))
        h_size = int((float(img.size[1]) * float(w_percent)))
        resized_img = img.resize((max_width, h_size), Image.Resampling.LANCZOS)
        resized_img.save(output_path, "JPEG", quality=85)