import torch
import cv2
import numpy as np
from craft import CRAFT
from craft_utils import getDetBoxes, adjustResultCoordinates
from parseq import SceneTextDataModule
from PIL import Image

# Load CRAFT model
def load_craft_model(craft_model_path):
    net = CRAFT()  # initialize CRAFT
    net.load_state_dict(torch.load(craft_model_path, map_location='cpu'))
    net.eval()
    return net

# Load PARSeq model
def load_parseq_model():
    parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
    img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)
    return parseq, img_transform

# Text detection using CRAFT
def detect_text(image, net):
    # Preprocess image
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, canvas_size=1280)
    ratio_h = ratio_w = 1 / target_ratio

    # Convert to PyTorch tensor and normalize
    x = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0)

    with torch.no_grad():
        y, _ = net(x)
    
    # Get bounding boxes
    boxes, polys = getDetBoxes(y[0, :, :, 0].cpu().numpy(), y[0, :, :, 1].cpu().numpy(), text_threshold=0.7, link_threshold=0.4)
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    return boxes

# Text recognition using PARSeq
def recognize_text(parseq, img_transform, image, boxes):
    results = []
    for box in boxes:
        x_min, y_min = np.min(box, axis=0).astype(int)
        x_max, y_max = np.max(box, axis=0).astype(int)
        cropped_img = image[y_min:y_max, x_min:x_max]

        # Convert image to PIL format for PARSeq
        pil_img = Image.fromarray(cropped_img).convert('RGB')
        pil_img = img_transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            logits = parseq(pil_img)
            pred = logits.softmax(-1)
            label, confidence = parseq.tokenizer.decode(pred)
            results.append((label[0], confidence))
    return results

# Main function
if __name__ == "__main__":
    # Load models
    craft_model_path = 'path/to/craft_model.pth'
    craft_net = load_craft_model(craft_model_path)
    parseq_net, parseq_transform = load_parseq_model()

    # Load image
    img_path = 'path/to/test_image.jpg'
    image = cv2.imread(img_path)

    # Detect text boxes using CRAFT
    text_boxes = detect_text(image, craft_net)

    # Recognize text using PARSeq
    recognized_texts = recognize_text(parseq_net, parseq_transform, image, text_boxes)

    # Print results
    for i, (text, confidence) in enumerate(recognized_texts):
        print(f"Text {i+1}: {text} (Confidence: {confidence:.2f})")
