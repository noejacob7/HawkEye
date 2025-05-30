import imageio
from PIL import Image, ImageDraw
import numpy as np
from detection.yolo_wrapper import YOLOv11Wrapper

# Initialize YOLO detector
yolo = YOLOv11Wrapper(conf_threshold=0.3)

# Load video
reader = imageio.get_reader("data/test/test_002.mp4")
fps = reader.get_meta_data()['fps']
size = reader.get_meta_data()['size']

# Prepare writer
writer = imageio.get_writer("output_tracked.mp4", fps=fps)

# Process frames
for frame in reader:
    # Convert to PIL image
    image = Image.fromarray(frame)
    frame_array = np.array(image)

    # Predict with YOLO
    boxes, _, _ = yolo.predict(frame_array)

    # Draw boxes
    draw = ImageDraw.Draw(image)
    for x, y, w, h in boxes:
        draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=3)

    # Write frame
    writer.append_data(np.array(image))

reader.close()
writer.close()
print("[✓] Detection-only video saved as output_tracked.mp4")
