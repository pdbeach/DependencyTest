To train an object detection model using DETR with the specified requirements, follow this structured approach:

### Step-by-Step Explanation

1. **Collect and Prepare Dataset**:
   - **Dataset Structure**: Images and corresponding LabelMe JSON annotations are stored in the same directory with matching filenames.
   - **Class Mapping**: Extract unique labels from all JSON files to create a mapping from label names to class IDs.

2. **Dataset Class**:
   - **UltrasoundDataset**: Loads images and annotations, applies transformations (resize, normalize), and converts bounding boxes to the required format.

3. **DETR Model**:
   - Load the pre-trained DETR model from torch.hub, adjusting the number of output classes based on the dataset.

4. **Loss Functions**:
   - **HungarianMatcher**: Matches predicted and ground truth boxes using the Hungarian algorithm.
   - **SetCriterion**: Computes the total loss combining classification, bounding box regression, and GIoU loss.

5. **Training Loop**:
   - Configure DataLoader with a custom collate function.
   - Set up optimizer and learning rate scheduler.
   - Iterate over epochs, compute losses, backpropagate, and update weights.

### Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
import numpy as np
from scipy.optimize import linear_sum_assignment

# Helper functions for box transformations and IoU
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    return torch.stack([x_c - 0.5 * w, y_c - 0.5 * h,
                        x_c + 0.5 * w, y_c + 0.5 * h], dim=-1)

def generalized_box_iou(boxes1, boxes2):
    max_xy = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    min_xy = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter = (max_xy - min_xy).clamp(min=0).prod(2)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - inter
    iou = inter / union
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    hull = (rb - lt).clamp(min=0).prod(2)
    return iou - (hull - union) / hull

# Hungarian Matcher for DETR loss
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_bbox=1, cost_giou=1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, n_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        tgt_ids = torch.cat([t["labels"] for t in targets])
        tgt_bbox = torch.cat([t["boxes"] for t in targets])
        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
        C = C.view(bs, n_queries, -1).cpu()
        sizes = [len(t["boxes"]) for t in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

# Loss criterion combining different components
class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {'loss_ce': loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none').sum() / num_boxes
        loss_giou = (1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))).sum() / num_boxes
        return {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        if 'aux_outputs' in outputs:
            for i, aux in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux, targets)
                l_dict = self.loss_labels(aux, targets, indices, num_boxes)
                l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
                l_dict = self.loss_boxes(aux, targets, indices, num_boxes)
                l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses

# Dataset class
class UltrasoundDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = []
        self.labels = set()
        valid_exts = {'.png', '.jpg', '.jpeg'}
        for fname in os.listdir(root_dir):
            if os.path.splitext(fname)[1].lower() in valid_exts:
                json_path = os.path.join(root_dir, os.path.splitext(fname)[0] + '.json')
                if os.path.exists(json_path):
                    self.image_files.append((os.path.join(root_dir, fname), json_path))
                    with open(json_path) as f:
                        data = json.load(f)
                        for shape in data['shapes']:
                            self.labels.add(shape['label'])
        self.labels = sorted(self.labels)
        self.label_to_id = {lbl: i for i, lbl in enumerate(self.labels)}
        self.transform = transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path, json_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        image = self.transform(image)
        with open(json_path) as f:
            data = json.load(f)
        boxes, labels = [], []
        for shape in data['shapes']:
            points = np.array(shape['points'])
            x0, y0 = points.min(0)
            x1, y1 = points.max(0)
            boxes.append([x0, y0, x1, y1])
            labels.append(self.label_to_id[shape['label']])
        scale_x = 800 / orig_w
        scale_y = 800 / orig_h
        boxes = torch.tensor(boxes, dtype=torch.float32) * torch.tensor([scale_x, scale_y, scale_x, scale_y])
        labels = torch.tensor(labels, dtype=torch.int64)
        return image, {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([idx])}

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = UltrasoundDataset('C:/Datasets/UltrasoundImages')
    num_classes = len(dataset.labels)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True, num_classes=num_classes)
    model.to(device)
    
    matcher = HungarianMatcher()
    weight_dict = {'loss_ce': 1, 'loss_bbox': 1, 'loss_giou': 1}
    criterion = SetCriterion(num_classes, matcher, weight_dict, eos_coef=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')
    torch.save(model.state_dict(), 'detr_trained.pth')

if __name__ == '__main__':
    main()
```

### Notes

- **Dataset Handling**: The dataset is loaded with images resized to 800x800, and bounding boxes are scaled accordingly.
- **Model Configuration**: The DETR model is loaded with the number of classes derived from the dataset.
- **Loss Calculation**: Utilizes a combination of classification, bounding box regression, and GIoU loss as per DETR's approach.
- **Training**: The model is trained for 10 epochs with an AdamW optimizer, and the trained weights are saved.

This code provides a complete pipeline for training DETR on a custom dataset with LabelMe annotations, handling image preprocessing, model configuration, and loss calculation.