
import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Path to your saved model weights
MODEL_PATH = "fasterrcnn_resnet50.pth"

# Path to a folder with test images or a single image
TEST_IMAGE_PATH = "path/to/test_image.jpg"  # or "path/to/directory"

# Label map must match what you used during training
# Example: If background=0, and you had classes like cat=1, dog=2
label_map = {
    0: "background",
    1: "cat",
    2: "dog"
    # Add other classes as per your training
}

def get_model(num_classes):
    # Load a model pre-trained on COCO
    model = torch.hub.load('pytorch/vision', 'fasterrcnn_resnet50_fpn', pretrained=False)
    # Replace the classifier with a new one (must match your training setup)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torch.nn.Linear(in_features, num_classes)
    return model

def load_model(model_path, num_classes):
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def predict(model, img_path, device):
    # Load and transform the image
    img = Image.open(img_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).to(device)

    # Run inference
    with torch.no_grad():
        predictions = model([img_tensor])
    return img, predictions[0]  # predictions[0] for single image

def visualize_predictions(img, predictions, label_map, score_thresh=0.5):
    # Unpack predictions
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()

    # Convert PIL to matplotlib
    fig, ax = plt.subplots(1, figsize=(12,9))
    ax.imshow(img)

    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin-5, f"{label_map[label]}: {score:.2f}",
                fontsize=12, color='yellow', bbox=dict(facecolor='red', alpha=0.5))
        
    plt.show()

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = len(label_map)  # Make sure this matches what you trained with

    # Load the trained model
    model = load_model(MODEL_PATH, num_classes).to(device)

    # If TEST_IMAGE_PATH is a directory, run inference on all images in it
    if os.path.isdir(TEST_IMAGE_PATH):
        for f in os.listdir(TEST_IMAGE_PATH):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(TEST_IMAGE_PATH, f)
                img, preds = predict(model, img_path, device)
                print(f"Predictions for {f}:")
                for box, label, score in zip(preds['boxes'], preds['labels'], preds['scores']):
                    if score > 0.5:
                        print(f"  Label: {label_map[label]}, Score: {score:.2f}, Box: {box.tolist()}")
                # Optional: visualize
                # visualize_predictions(img, preds, label_map)
    else:
        # Single image prediction
        img, preds = predict(model, TEST_IMAGE_PATH, device)
        print(f"Predictions for {TEST_IMAGE_PATH}:")
        for box, label, score in zip(preds['boxes'], preds['labels'], preds['scores']):
            if score > 0.5:
                print(f"  Label: {label_map[label]}, Score: {score:.2f}, Box: {box.tolist()}")
        # Optional: visualize
        # visualize_predictions(img, preds, label_map)