import torch
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from torchvision import transforms

model_path = "fasterrcnn_resnet50.pth"
test_image = "path/to/test_image.jpg"
num_classes = 3  # include background

# Create the model and replace the predictor just as you did during training
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

img = Image.open(test_image).convert("RGB")
img_tensor = transforms.ToTensor()(img)

with torch.no_grad():
    preds = model([img_tensor])[0]

print(preds)