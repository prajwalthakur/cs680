# %%
import numpy as np
import pandas as pd
import imageio.v3 as imageio
import albumentations as A

# import torch_xla as xla
# import torch_xla.core.xla_model as xm
# import torch_xla.distributed.xla_multiprocessing as xmp
# import torch_xla.distributed.xla_backend

from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from tqdm.notebook import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


import torch
import timm
import torchmetrics
import time
import psutil
import cv2

# %%
torch.manual_seed(42)
np.random.seed(42)
device = 'cuda' #xla.device()

# %%


# %%
class Config():
    IMAGE_SIZE = 128
    TARGET_IMAGE_SIZE = 224
    BACKBONE = 'swin_large_patch4_window12_384.ms_in22k_ft_in1k'
    TARGET_COLUMNS =['X4_mean', 'X11_mean', 'X18_mean', 'X26_mean', 'X50_mean', 'X3112_mean' ]
    N_TARGETS = len(TARGET_COLUMNS)
    BATCH_SIZE = 256
    LR_MAX = 1e-4
    WEIGHT_DECAY = 0.01
    N_EPOCHS = 5
    TRAIN_MODEL = True
    IS_INTERACTIVE = True  # os.environ['KAGGLE_KERNEL_RUN_TYPE'] == 'Interactive'
    tpu_ids = range(8)
    Lower_Quantile = 0.005
    Upper_Quantile = 0.980
    SHRINK_SAMPLES = False
    WANDB_INIT = True
CONFIG = Config()

# %%
import wandb
if CONFIG.WANDB_INIT:
    wandb.login()
    wandb.init(project="cs680v3",group="swin_tf",name="submission_internet_script_swin_with_table_large_224",
            config = {
        "LR_max": CONFIG.LR_MAX,
        "WEIGHT_DECAY":CONFIG.WEIGHT_DECAY,
        "train_batch" : CONFIG.BATCH_SIZE
        })

# %%
class TrainDataset(Dataset):
    def __init__(self, X_jpeg_bytes, X_tabular, y, transforms=None):
        self.X_jpeg_bytes = X_jpeg_bytes
        self.X_tabular = X_tabular
        self.y = y
        self.transforms = transforms

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        X_sample = self.transforms(
            image=imageio.imread(self.X_jpeg_bytes[index]),
        )['image']
        X_tabular_sample = self.X_tabular[index]
        y_sample = self.y[index]

        return X_sample, X_tabular_sample, y_sample

# %%
class TabularBackbone(nn.Module):
    def __init__(self, n_features, out_features):
        super().__init__()
        self.out_features = out_features
        self.fc = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            # nn.Dropout(0.1),
            nn.Linear(512, out_features),
        )

    def forward(self, x):
        return self.fc(x)
    
class ImageBackbone(nn.Module):
    def __init__(self, backbone_name, weight_path, out_features, fixed_feature_extractor=False):
        super().__init__()
        self.out_features = out_features
        self.backbone =timm.create_model('swin_large_patch4_window7_224.ms_in22k', pretrained=False, num_classes=CONFIG.N_TARGETS) # timm.create_model('swin_large_patch4_window12_384.ms_in22k_ft_in1k', pretrained=True, num_classes=CONFIG.N_TARGETS)
        swin_fine_tuned_weight = torch.load("/home/prajwal/cs680/cs680_kaggle/data/swin_large_fine_tuning_train.pth")
        swin_fine_tuned_weight  = {key.replace("img_backbone.backbone.", ""): value for key, value in swin_fine_tuned_weight.items()}
        self.backbone.load_state_dict(swin_fine_tuned_weight)
        if fixed_feature_extractor:
            for param in self.backbone.parameters():
                param.requires_grad = False
        in_features = self.backbone.num_features
        
        self.backbone.head = nn.Identity()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, out_features),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.permute(0, 3, 1, 2)
        return self.head(x)

class Model(nn.Module):
    def __init__(self, img_backbone, tab_backbone, out_features:int):
        super().__init__()
        self.img_backbone = img_backbone
        self.tab_backbone = tab_backbone
        self.fc = nn.Sequential(
            nn.Linear(self.tab_backbone.out_features + self.img_backbone.out_features, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            # nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            # nn.Dropout(0.1),
            nn.Linear(256, out_features),
        )

    def forward(self, img, tab):
        img_features = self.img_backbone(img)
        tab_features = self.tab_backbone(tab)
        features = torch.cat([img_features, tab_features], dim=1)
        return self.fc(features)

# %%
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.sum += val.sum()
        self.count += val.numel()
        self.avg = self.sum / self.count

# %%
def get_lr_scheduler(optimizer):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=CONFIG.LR_MAX,
        total_steps=CONFIG.N_STEPS,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=1e1,
        final_div_factor=1e1,
    )

# %%
import os
BASE_DIR = os.path.join(os.getcwd() , 'data')
train = pd.read_csv(BASE_DIR  +  '/train.csv')
test =  pd.read_csv(BASE_DIR  +  '/test.csv')

# %%
for column in CONFIG.TARGET_COLUMNS:
    lower_quantile = train[column].quantile(CONFIG.Lower_Quantile)
    upper_quantile = train[column].quantile(CONFIG.Upper_Quantile)
    train = train[(train[column] >= lower_quantile) & (train[column] <= upper_quantile)]

tabular = train.drop(columns = ['id'] + CONFIG.TARGET_COLUMNS)
test_tabular = test.drop(columns = ['id'])

# %%
LOG_FEATURES = ['X4_mean', 'X11_mean', 'X18_mean', 'X26_mean', 'X50_mean', 'X3112_mean' ]

y_train = np.zeros_like(train[CONFIG.TARGET_COLUMNS], dtype=np.float32)
for target_idx, target in enumerate(CONFIG.TARGET_COLUMNS):
    v = train[target].values
    if target in LOG_FEATURES:
        v = np.log10(v)
    y_train[:, target_idx] = v

# %%
# normalize tabular inputs
X_SCALER = StandardScaler()
tabular_scaled = X_SCALER.fit_transform(tabular).astype(np.float32)
test_tabular_scaled = X_SCALER.transform(test_tabular).astype(np.float32)

Y_SCALER = StandardScaler()
y_train_scaled = Y_SCALER.fit_transform(y_train).astype(np.float32)

# %%
print('JPEG Files Processing:')
train['file_path'] = train['id'].apply(lambda s: f'{BASE_DIR}/train_images/{s}.jpeg')
train['jpeg_bytes'] = train['file_path'].apply(lambda fp: open(fp, 'rb').read())


test['file_path'] = test['id'].apply(lambda s: f'{BASE_DIR}/test_images/{s}.jpeg')
test['jpeg_bytes'] = test['file_path'].apply(lambda fp: open(fp, 'rb').read())
print('JPEG Files Processing End')  

# %%
CONFIG.N_TRAIN_SAMPLES = len(tabular_scaled)
CONFIG.N_STEPS_PER_EPOCH = (CONFIG.N_TRAIN_SAMPLES // CONFIG.BATCH_SIZE)
CONFIG.N_STEPS = CONFIG.N_STEPS_PER_EPOCH * CONFIG.N_EPOCHS + 1

# %%
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS = A.Compose([
    A.HorizontalFlip(p=0.25),
    A.RandomResizedCrop(size=(CONFIG.TARGET_IMAGE_SIZE,CONFIG.TARGET_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC ,p=0.5),  # Simulate different crops
    A.Resize(CONFIG.TARGET_IMAGE_SIZE,CONFIG.TARGET_IMAGE_SIZE),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.25),
    A.ImageCompression(quality_lower=85, quality_upper=100, p=0.25), 
    #A.GaussianBlur(blur_limit=(3, 7), p=0.5),  # Introduce slight blur
    A.ToFloat(),
    A.Normalize(mean=MEAN, std=STD, max_pixel_value=1),
    ToTensorV2(),
])

TEST_TRANSFORMS = A.Compose([
    A.Resize(CONFIG.TARGET_IMAGE_SIZE,CONFIG.TARGET_IMAGE_SIZE),
    #A.CenterCrop(CONFIG.TARGET_IMAGE_SIZE,CONFIG.TARGET_IMAGE_SIZE),
    A.ToFloat(),
    A.Normalize(mean=MEAN, std=STD, max_pixel_value=1),
    ToTensorV2(),
])

# %%
# # train / test split
# from sklearn.model_selection import train_test_split
# train_df , test_df = train_test_split(train,test_size=0.1,shuffle=True)

train_idx =np.random.choice(len(train), int(0.9 * len(train)), replace=False)
test_idx = np.setdiff1d(np.arange(len(train)), train_idx)

train_images = train['jpeg_bytes'].values[train_idx]
train_tabular = tabular_scaled[train_idx]
train_y = y_train_scaled[train_idx]

val_images = train['jpeg_bytes'].values[test_idx]
val_tabular = tabular_scaled[test_idx]
val_y = y_train_scaled[test_idx]

test_images = test['jpeg_bytes'].values
test_tabular = test_tabular_scaled

# %%
train_dataset = TrainDataset(
    train_images,
    train_tabular,
    train_y,
    TRAIN_TRANSFORMS
)

validation_dataset = TrainDataset(
    val_images,
    val_tabular,
    val_y,
    TEST_TRANSFORMS
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=CONFIG.BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=0#psutil.cpu_count(),
)

validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=CONFIG.BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    num_workers=0#psutil.cpu_count(),
)

test_dataset = TrainDataset(
    test['jpeg_bytes'].values,
    test_tabular,
    test['id'].values,
    TEST_TRANSFORMS,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    drop_last=False,
    num_workers=0#psutil.cpu_count(),
)

# %%
img_backbone = ImageBackbone('swin_large_patch4_window12_384.ms_in22k_ft_in1k', '/kaggle/input/swin-transformer-v1-planttraits2024-finetuned/pytorch/log3-noval-8epoch/1/model_08.pth', 384, fixed_feature_extractor=True)
tab_backbone = TabularBackbone(n_features=tabular_scaled.shape[1], out_features=128)

model = Model(img_backbone, tab_backbone, CONFIG.N_TARGETS)
model = model.to(device)

# %%
MAE = torchmetrics.regression.MeanAbsoluteError().to(device)
R2 = torchmetrics.regression.R2Score(num_outputs=CONFIG.N_TARGETS, multioutput='uniform_average').to(device)
LOSS = AverageMeter()

Y_MEAN = torch.tensor(y_train).mean(dim=0).to(device)
EPS = torch.tensor([1e-6]).to(device)

# %%
LOSS_FN = nn.SmoothL1Loss()  # r2_loss

optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=CONFIG.LR_MAX,
    weight_decay=CONFIG.WEIGHT_DECAY,
)

LR_SCHEDULER = get_lr_scheduler(optimizer)

# %%
class BestModelSaveCallback:
    def __init__(self, save_path):
        self.save_path = save_path
        self.best_accuracy = -1

    def __call__(self, accuracy,model):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            model.to(device = "cpu")
            torch.save(model.state_dict(), self.save_path)
            model.to(device='cuda')
            
            print("generating result on test data")
            SUBMISSION_ROWS = []
            model.eval()

            for X_image, X_tabular, test_id in tqdm(test_dataloader):
                with torch.no_grad():
                    y_pred = model(X_image.to(device), X_tabular.to(device)).detach().cpu().numpy()
                
                y_pred = Y_SCALER.inverse_transform(y_pred).squeeze()
                row = {'id': int(test_id)}
                
                for k, v in zip(CONFIG.TARGET_COLUMNS, y_pred):
                    if k in LOG_FEATURES:
                        row[k.replace('_mean', '')] = 10 ** v
                    else:
                        row[k.replace('_mean', '')] = v

                SUBMISSION_ROWS.append(row)
                
            submission_df = pd.DataFrame(SUBMISSION_ROWS)
            submission_df.to_csv('internet_script_self_model.csv', index=False)
            print("Submit!")
MODEL_NAME_SAVE = 'submission_internet_Script_self_swin224.pth'
best_model_callback = BestModelSaveCallback(save_path=os.path.join(BASE_DIR,MODEL_NAME_SAVE))

# %%
print("Start Training:")
for epoch in range(CONFIG.N_EPOCHS):
    if CONFIG.WANDB_INIT :
        wandb.log({"epoch":epoch })
    MAE.reset()
    R2.reset()
    LOSS.reset()
    model.train()

    for step, (X_image, X_tabular, y_true) in enumerate(train_dataloader):
        X_image = X_image.to(device)
        X_tabular = X_tabular.to(device)
        y_true = y_true.to(device)
        model = model.to(device)
        t_start = time.perf_counter_ns()
        y_pred = model(X_image, X_tabular)
        loss = LOSS_FN(y_pred, y_true)
        LOSS.update(loss)
        loss.backward()
        optimizer.step()
        # xm.optimizer_step(optimizer, barrier=True)
        optimizer.zero_grad()
        LR_SCHEDULER.step()
        MAE.update(y_pred, y_true)
        R2.update(y_pred, y_true)
        if CONFIG.WANDB_INIT:
            wandb.log({"Training-Loss":LOSS.avg.detach().cpu().item()  , "Training-MAE" :  MAE.compute().item() , "Training-R2":  R2.compute().item() })
        # if not CONFIG.IS_INTERACTIVE and (step + 1) == CONFIG.N_STEPS_PER_EPOCH:
        #     print(
        #         f'EPOCH {epoch + 1:02d}, {step + 1:04d}/{CONFIG.N_STEPS_PER_EPOCH} | ' +
        #         f'loss: {LOSS.avg:.4f}, mae: {MAE.compute().item():.4f}, r2: {R2.compute().item():.4f}, ' +
        #         f'step: {(time.perf_counter_ns() - t_start) * 1e-9:.3f}s, lr: {LR_SCHEDULER.get_last_lr()[0]:.2e}',
        #     )
        # elif CONFIG.IS_INTERACTIVE:
        #     print(
        #         f'\rEPOCH {epoch + 1:02d}, {step + 1:04d}/{CONFIG.N_STEPS_PER_EPOCH} | ' +
        #         f'loss: {LOSS.avg:.4f}, mae: {MAE.compute().item():.4f}, r2: {R2.compute().item():.4f}, ' +
        #         f'step: {(time.perf_counter_ns() - t_start) * 1e-9:.3f}s, lr: {LR_SCHEDULER.get_last_lr()[0]:.2e}',
        #         end='\n' if (step + 1) == CONFIG.N_STEPS_PER_EPOCH else '', flush=True,
        #     )
    model = model.to(device)
    model.eval()
    MAE.reset()
    R2.reset()
    LOSS.reset()

    print('in  Validation:')
    with torch.no_grad():
        for X_image, X_tabular, y_true in (validation_dataloader):
            X_image = X_image.to(device)
            y_true = y_true.to(device)
            X_tabular = X_tabular.to(device)
            y_pred = model(X_image, X_tabular)
            loss = LOSS_FN(y_pred, y_true)
            LOSS.update(loss)
            MAE.update(y_pred, y_true)
            R2.update(y_pred, y_true)

            # if not CONFIG.IS_INTERACTIVE:
            #     print(
            #         f'EPOCH {epoch + 1:02d}, VALIDATION | ' +
            #         f'loss: {LOSS.avg:.4f}, mae: {MAE.compute().item():.4f}, r2: {R2.compute().item():.4f}',
            #     )
            # elif CONFIG.IS_INTERACTIVE:
            #     print(
            #         f'\rEPOCH {epoch + 1:02d}, VALIDATION | ' +
            #         f'loss: {LOSS.avg:.4f}, mae: {MAE.compute().item():.4f}, r2: {R2.compute().item():.4f}',
            #         end='\n',
            #     )
        if CONFIG.WANDB_INIT:
            wandb.log({"Validation-Loss":LOSS.avg.detach().cpu().item()  , "Validation-MAE" :  MAE.compute().item() , "Validation-R2":  R2.compute().item() })
    best_model_callback(R2.compute().item(),model)

# %%
torch.save(model.to('cpu').state_dict(), os.path.join(BASE_DIR,MODEL_NAME_SAVE))

# %%
# load model
model.to(device)

SUBMISSION_ROWS = []
model.eval()

for X_image, X_tabular, test_id in tqdm(test_dataloader):
    with torch.no_grad():
        y_pred = model(X_image.to(device), X_tabular.to(device)).detach().cpu().numpy()
    
    y_pred = Y_SCALER.inverse_transform(y_pred).squeeze()
    row = {'id': int(test_id)}
    
    for k, v in zip(CONFIG.TARGET_COLUMNS, y_pred):
        if k in LOG_FEATURES:
            row[k.replace('_mean', '')] = 10 ** v
        else:
            row[k.replace('_mean', '')] = v

    SUBMISSION_ROWS.append(row)
    
submission_df = pd.DataFrame(SUBMISSION_ROWS)
submission_df.to_csv('internet_script_self_model_final.csv', index=False)
print("Submit!")


