
import torch
from PIL import Image
from torchvision import transforms, models

model_path = "fasterrcnn_resnet50.pth"  # Your saved model
test_image = "path/to/test_image.jpg"    # Your test image
num_classes = 3                          # Adjust based on your training (background + your classes)

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torch.nn.Linear(in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

img = Image.open(test_image).convert("RGB")
img_tensor = transforms.ToTensor()(img)

with torch.no_grad():
    preds = model([img_tensor])[0]

print(preds)  # Prints keys: 'boxes', 'labels', 'scores'