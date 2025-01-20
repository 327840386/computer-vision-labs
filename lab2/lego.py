
# 2. 数据准备
import os
import random
import shutil
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torchvision
from torchvision import transforms
import torchvision.datasets as dset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.transforms as T
from PIL import Image
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import cv2

# 假设你已经将数据集下载到名为 'lego_dataset' 的文件夹中
DATASET_PATH = '/Users/luyuhao/Desktop/computer_vision/lego_dataset'
IMAGES_PATH = os.path.join(DATASET_PATH, 'images')
ANNOTATIONS_PATH = os.path.join(DATASET_PATH, 'annotations')

# 创建训练、验证和测试集的文件夹
os.makedirs('data/images/train', exist_ok=True)
os.makedirs('data/images/val', exist_ok=True)
os.makedirs('data/images/test', exist_ok=True)
os.makedirs('data/labels/train', exist_ok=True)
os.makedirs('data/labels/val', exist_ok=True)
os.makedirs('data/labels/test', exist_ok=True)

# 划分数据集
images = [f for f in os.listdir(IMAGES_PATH) if f.endswith('.jpg')]
train_imgs, val_test_imgs = train_test_split(images, test_size=0.3, random_state=42)
val_imgs, test_imgs = train_test_split(val_test_imgs, test_size=0.5, random_state=42)

# 移动图片到相应的文件夹

# 2. 数据准备
import os
import random
import shutil
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torchvision
from torchvision import transforms
import torchvision.datasets as dset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.transforms as T
from PIL import Image
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import cv2

# 假设你已经将数据集下载到名为 'lego_dataset' 的文件夹中
DATASET_PATH = '/Users/luyuhao/Desktop/computer_vision/lego_dataset'
IMAGES_PATH = os.path.join(DATASET_PATH, 'images')
ANNOTATIONS_PATH = os.path.join(DATASET_PATH, 'annotations')

# 创建训练、验证和测试集的文件夹
os.makedirs('data/images/train', exist_ok=True)
os.makedirs('data/images/val', exist_ok=True)
os.makedirs('data/images/test', exist_ok=True)
os.makedirs('data/labels/train', exist_ok=True)
os.makedirs('data/labels/val', exist_ok=True)
os.makedirs('data/labels/test', exist_ok=True)

# 划分数据集并确保图像正确注释
images = [f for f in os.listdir(IMAGES_PATH) if f.endswith('.jpg')]
valid_images = []
for image in images:
    label_path = os.path.join(ANNOTATIONS_PATH, image.replace('.jpg', '.xml'))
    if os.path.exists(label_path):
        valid_images.append(image)

# 只使用部分数据以减少数据量
sample_size = min(200, len(valid_images))  # 使用最多 600 张图像
sampled_images = random.sample(valid_images, sample_size)

train_imgs, val_test_imgs = train_test_split(sampled_images, test_size=0.3, random_state=42)
val_imgs, test_imgs = train_test_split(val_test_imgs, test_size=0.5, random_state=42)

# 移动图片到相应的文件夹
def move_files(file_list, source_folder, dest_folder):
    for file_name in file_list:
        shutil.copy(os.path.join(source_folder, file_name), dest_folder)

move_files(train_imgs, IMAGES_PATH, 'data/images/train')
move_files(val_imgs, IMAGES_PATH, 'data/images/val')
move_files(test_imgs, IMAGES_PATH, 'data/images/test')

# 3. 数据集加载器
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

class LegoDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(root)))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        label_path = os.path.join(ANNOTATIONS_PATH, self.imgs[idx].replace('.jpg', '.xml'))
        img = Image.open(img_path).convert("RGB")
        boxes = []
        tree = ET.parse(label_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            x_min = float(bbox.find('xmin').text)
            y_min = float(bbox.find('ymin').text)
            x_max = float(bbox.find('xmax').text)
            y_max = float(bbox.find('ymax').text)
            boxes.append([x_min, y_min, x_max, y_max])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64) * 1  # 所有对象标记为 'lego'
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        masks = torch.zeros((len(boxes), img.height, img.width), dtype=torch.uint8)
        target = {"boxes": boxes, "labels": labels, "masks": masks, "image_id": image_id, "area": area, "iscrowd": iscrowd}
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.imgs)

# 4. 定义模型
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
model.roi_heads.box_predictor.cls_score = torch.nn.Linear(1024, 2)  # 修改为 2 类（背景和 LEGO）

# 5. 训练设置
def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == "__main__":
    # 使用 GPU（如果可用）
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # 数据加载器
    train_dataset = LegoDataset('data/images/train', get_transform(train=True))
    val_dataset = LegoDataset('data/images/val', get_transform(train=False))
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # 6. 模型训练
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # 学习率调整器
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 训练过程
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        i = 0
        for images, targets in train_data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        lr_scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {losses.item()}")

    # 7. 模型评估
    def evaluate(model, data_loader, device):
        model.eval()
        with torch.no_grad():
            for images, targets in data_loader:
                images = list(image.to(device) for image in images)
                outputs = model(images)
                # 输出评估结果（可以根据需要进行进一步处理和显示）

    evaluate(model, val_data_loader, device)

    # 8. 推断
    model.eval()
    img_path = 'data/images/test/example.jpg'
    img = Image.open(img_path).convert("RGB")
    img_tensor = get_transform(train=False)(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(img_tensor)

    # 可视化推断结果
    img_cv = cv2.imread(img_path)
    for box in prediction[0]['boxes']:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    plt.show()
