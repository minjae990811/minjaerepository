import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms

# Placeholder dataset loader
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.images = list(self.images_dir.glob('*'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks_dir / img_path.name
        image = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        mask = transforms.ToTensor()(Image.open(mask_path).convert('L'))
        if self.transform:
            image = self.transform(image)
        return image, mask

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune SAM2 model')
    parser.add_argument('--data_path', type=str, required=True, help='Dataset root path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Pretrained SAM2 checkpoint')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    return parser.parse_args()

def main():
    args = parse_args()

    train_dataset = SegmentationDataset(
        Path(args.data_path)/'train'/ 'images',
        Path(args.data_path)/'train'/ 'masks'
    )

    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Load pretrained model
    model = torch.hub.load('facebookresearch/segment-anything', 'sam2', source='github')
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCELoss()

    for epoch in range(args.epochs):
        for images, masks in dataloader:
            preds = model(images)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} / {args.epochs} - Loss: {loss.item():.4f}")

    Path('checkpoints').mkdir(exist_ok=True)
    torch.save({'model': model.state_dict()}, f'checkpoints/sam2_finetuned.pt')

if __name__ == '__main__':
    main()
