# import os
# from yolact import Yolact  # ensure yolact repository is in PYTHONPATH
# from data import COCODetection
# from utils.config import cfg, set_cfg

# set_cfg('config/yolact_plus_config.py')
# model = Yolact()
# model.load_weights('weights/yolact_plus_base.pth')

# train_dataset = COCODetection('data/train/images', 'data/train/annotations.json', transform=None)
# val_dataset   = COCODetection('data/valid/images', 'data/valid/annotations.json', transform=None)

# model.train_loop(
#     train_dataset,
#     val_dataset,
#     batch_size=16,
#     epochs=100,
#     save_folder='results/yolactpp'
# )
