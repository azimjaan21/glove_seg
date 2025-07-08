# import torch
# from torch.utils.data import DataLoader
# from torchvision.datasets import CocoDetection
# import torchvision.transforms as T
# from autodistill_efficientsam import EfficientSAM

# transform = T.Compose([T.ToTensor()])
# train_ds = CocoDetection('data/train/images', 'data/train/annotations.json', transform)
# val_ds   = CocoDetection('data/valid/images', 'data/valid/annotations.json', transform)
# train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
# val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=4)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = EfficientSAM(None).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# for epoch in range(1, 51):
#     model.train()
#     for imgs, anns in train_loader:
#         imgs = imgs.to(device)
#         masks = [a['segmentation'] for a in anns]
#         loss = model.train_step(imgs, masks)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     # Optionally validate here

# torch.save(model.state_dict(), 'results/efficientsam/efficient_sam_finetuned.pth')
