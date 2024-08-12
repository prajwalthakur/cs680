# %%
# %%

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import IPython
from glob import glob
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
import tqdm
#import seaborn as sns
import albumentations as A
import torch
import torch.nn as nn
import timm
import torcheval 
import wandb
import torchvision
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import seaborn as sns
from albumentations.pytorch import ToTensorV2
plt.style.use('ggplot')
plt.rcParams.update(**{'figure.dpi':150})
import psutil
import imageio.v3 as imageio
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler , MinMaxScaler
import scipy as sp
# %%
RANDOM_NUMBER = 42
torch.manual_seed(RANDOM_NUMBER)

# %% [markdown]
# # select Device

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else  "cpu") #"cuda:1" if torch.cuda.is_available() else 
# torch.cuda.set_per_process_memory_fraction(0.95, device=DEVICE)
torch.cuda.empty_cache()

# %%


# %%
class Config():

    BASE_DIR = os.path.join(os.getcwd() , 'data')
    train_df = pd.read_csv(BASE_DIR  +  '/train.csv')
    TRAIN_VAL_SPLIT_SIZE = 0.06
    TRAIN_BATCH_SIZE = 16
    VAL_BATCH_SIZE  =  16
    TEST_BATCH_SIZE =  2
    LR_MAX = 1e-4
    NUM_EPOCHS = 14
    TIM_NUM_CLASS =6 # 
    NORMALIZE_TARGET = "log_transform_mean_std"   #"log_transform" #
    RANDOM_NUMBER = 42
    NUM_FLODS  = 5
    NUM_CLASSES = 6
    TRAITS_NAME = ['X4_mean', 'X11_mean', 'X18_mean', 'X26_mean', 'X50_mean', 'X3112_mean' ]
    FOLD = 0 # Which fold to set as validation data
    IMAGE_SIZE =128
    TARGET_IMAGE_SIZE =  224
    T_MAX =        9
    LR_MODE = "step" # LR scheduler mode from one of "cos", "step", "exp"
    torch.manual_seed(RANDOM_NUMBER)
    INCLUDE_EXTRA_FEATURES = True
    EXTRA_FEATURES_NORMALIZATION = "standard_scalar"  #"min_max_normalization"  #
    WEIGHT_DECAY = 0.01
    TABULAR_NN_OUTPUT  = 256
    TIM_MODEL_NAME = "swinV2" #"swin_large" #"efficientnet_v2" # 
    TIMM_FINED_TUNED_WEIGHT = f'{BASE_DIR}/model_08_ensemble.pth'
    Lower_Quantile = 0.005
    Upper_Quantile = 0.98 #0.985
    WANDB_INIT = True
    # use XGBBOOST to find prominant features
    EXTRA_COLOUMN = ['WORLDCLIM_BIO1_annual_mean_temperature',
       'WORLDCLIM_BIO12_annual_precipitation',
       'WORLDCLIM_BIO13.BIO14_delta_precipitation_of_wettest_and_dryest_month',
       'WORLDCLIM_BIO15_precipitation_seasonality',
       'WORLDCLIM_BIO4_temperature_seasonality',
       'WORLDCLIM_BIO7_temperature_annual_range',
       'SOIL_bdod_0.5cm_mean_0.01_deg',
       'SOIL_bdod_100.200cm_mean_0.01_deg',
       'SOIL_bdod_15.30cm_mean_0.01_deg',
       'SOIL_bdod_30.60cm_mean_0.01_deg',
       'SOIL_bdod_5.15cm_mean_0.01_deg',
       'SOIL_bdod_60.100cm_mean_0.01_deg', 'SOIL_cec_0.5cm_mean_0.01_deg',
       'SOIL_cec_100.200cm_mean_0.01_deg',
       'SOIL_cec_15.30cm_mean_0.01_deg', 'SOIL_cec_30.60cm_mean_0.01_deg',
       'SOIL_cec_5.15cm_mean_0.01_deg', 'SOIL_cec_60.100cm_mean_0.01_deg',
       'SOIL_cfvo_0.5cm_mean_0.01_deg',
       'SOIL_cfvo_100.200cm_mean_0.01_deg',
       'SOIL_cfvo_15.30cm_mean_0.01_deg',
       'SOIL_cfvo_30.60cm_mean_0.01_deg',
       'SOIL_cfvo_5.15cm_mean_0.01_deg',
       'SOIL_cfvo_60.100cm_mean_0.01_deg',
       'SOIL_clay_0.5cm_mean_0.01_deg',
       'SOIL_clay_100.200cm_mean_0.01_deg',
       'SOIL_clay_15.30cm_mean_0.01_deg',
       'SOIL_clay_30.60cm_mean_0.01_deg',
       'SOIL_clay_5.15cm_mean_0.01_deg',
       'SOIL_clay_60.100cm_mean_0.01_deg',
       'SOIL_nitrogen_0.5cm_mean_0.01_deg',
       'SOIL_nitrogen_100.200cm_mean_0.01_deg',
       'SOIL_nitrogen_15.30cm_mean_0.01_deg',
       'SOIL_nitrogen_30.60cm_mean_0.01_deg',
       'SOIL_nitrogen_5.15cm_mean_0.01_deg',
       'SOIL_nitrogen_60.100cm_mean_0.01_deg',
       'SOIL_ocd_0.5cm_mean_0.01_deg', 'SOIL_ocd_100.200cm_mean_0.01_deg',
       'SOIL_ocd_15.30cm_mean_0.01_deg', 'SOIL_ocd_30.60cm_mean_0.01_deg',
       'SOIL_ocd_5.15cm_mean_0.01_deg', 'SOIL_ocd_60.100cm_mean_0.01_deg',
       'SOIL_ocs_0.30cm_mean_0.01_deg', 'SOIL_phh2o_0.5cm_mean_0.01_deg',
       'SOIL_phh2o_100.200cm_mean_0.01_deg',
       'SOIL_phh2o_15.30cm_mean_0.01_deg',
       'SOIL_phh2o_30.60cm_mean_0.01_deg',
       'SOIL_phh2o_5.15cm_mean_0.01_deg',
       'SOIL_phh2o_60.100cm_mean_0.01_deg',
       'SOIL_sand_0.5cm_mean_0.01_deg',
       'SOIL_sand_100.200cm_mean_0.01_deg',
       'SOIL_sand_15.30cm_mean_0.01_deg',
       'SOIL_sand_30.60cm_mean_0.01_deg',
       'SOIL_sand_5.15cm_mean_0.01_deg',
       'SOIL_sand_60.100cm_mean_0.01_deg',
       'SOIL_silt_0.5cm_mean_0.01_deg',
       'SOIL_silt_100.200cm_mean_0.01_deg',
       'SOIL_silt_15.30cm_mean_0.01_deg',
       'SOIL_silt_30.60cm_mean_0.01_deg',
       'SOIL_silt_5.15cm_mean_0.01_deg',
       'SOIL_silt_60.100cm_mean_0.01_deg', 'SOIL_soc_0.5cm_mean_0.01_deg',
       'SOIL_soc_100.200cm_mean_0.01_deg',
       'SOIL_soc_15.30cm_mean_0.01_deg', 'SOIL_soc_30.60cm_mean_0.01_deg',
       'SOIL_soc_5.15cm_mean_0.01_deg', 'SOIL_soc_60.100cm_mean_0.01_deg',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_01_._month_m1',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_02_._month_m1',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_03_._month_m1',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_04_._month_m1',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_05_._month_m1',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_01_._month_m10',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_02_._month_m10',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_03_._month_m10',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_04_._month_m10',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_05_._month_m10',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_01_._month_m11',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_02_._month_m11',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_03_._month_m11',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_04_._month_m11',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_05_._month_m11',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_01_._month_m12',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_02_._month_m12',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_03_._month_m12',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_04_._month_m12',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_05_._month_m12',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_01_._month_m2',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_02_._month_m2',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_03_._month_m2',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_04_._month_m2',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_05_._month_m2',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_01_._month_m3',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_02_._month_m3',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_03_._month_m3',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_04_._month_m3',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_05_._month_m3',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_01_._month_m4',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_02_._month_m4',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_03_._month_m4',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_04_._month_m4',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_05_._month_m4',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_01_._month_m5',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_02_._month_m5',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_03_._month_m5',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_04_._month_m5',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_05_._month_m5',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_01_._month_m6',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_02_._month_m6',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_03_._month_m6',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_04_._month_m6',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_05_._month_m6',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_01_._month_m7',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_02_._month_m7',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_03_._month_m7',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_04_._month_m7',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_05_._month_m7',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_01_._month_m8',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_02_._month_m8',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_03_._month_m8',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_04_._month_m8',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_05_._month_m8',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_01_._month_m9',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_02_._month_m9',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_03_._month_m9',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_04_._month_m9',
       'MODIS_2000.2020_monthly_mean_surface_reflectance_band_05_._month_m9',
       'VOD_C_2002_2018_multiyear_mean_m01',
       'VOD_C_2002_2018_multiyear_mean_m02',
       'VOD_C_2002_2018_multiyear_mean_m03',
       'VOD_C_2002_2018_multiyear_mean_m04',
       'VOD_C_2002_2018_multiyear_mean_m05',
       'VOD_C_2002_2018_multiyear_mean_m06',
       'VOD_C_2002_2018_multiyear_mean_m07',
       'VOD_C_2002_2018_multiyear_mean_m08',
       'VOD_C_2002_2018_multiyear_mean_m09',
       'VOD_C_2002_2018_multiyear_mean_m10',
       'VOD_C_2002_2018_multiyear_mean_m11',
       'VOD_C_2002_2018_multiyear_mean_m12',
       'VOD_Ku_1987_2017_multiyear_mean_m01',
       'VOD_Ku_1987_2017_multiyear_mean_m02',
       'VOD_Ku_1987_2017_multiyear_mean_m03',
       'VOD_Ku_1987_2017_multiyear_mean_m04',
       'VOD_Ku_1987_2017_multiyear_mean_m05',
       'VOD_Ku_1987_2017_multiyear_mean_m06',
       'VOD_Ku_1987_2017_multiyear_mean_m07',
       'VOD_Ku_1987_2017_multiyear_mean_m08',
       'VOD_Ku_1987_2017_multiyear_mean_m09',
       'VOD_Ku_1987_2017_multiyear_mean_m10',
       'VOD_Ku_1987_2017_multiyear_mean_m11',
       'VOD_Ku_1987_2017_multiyear_mean_m12',
       'VOD_X_1997_2018_multiyear_mean_m01',
       'VOD_X_1997_2018_multiyear_mean_m02',
       'VOD_X_1997_2018_multiyear_mean_m03',
       'VOD_X_1997_2018_multiyear_mean_m04',
       'VOD_X_1997_2018_multiyear_mean_m05',
       'VOD_X_1997_2018_multiyear_mean_m06',
       'VOD_X_1997_2018_multiyear_mean_m07',
       'VOD_X_1997_2018_multiyear_mean_m08',
       'VOD_X_1997_2018_multiyear_mean_m09',
       'VOD_X_1997_2018_multiyear_mean_m10',
       'VOD_X_1997_2018_multiyear_mean_m11',
       'VOD_X_1997_2018_multiyear_mean_m12'
       ]
    N_TARGETS  =len(TRAITS_NAME)  

CONFIG = Config()



# %%
if CONFIG.WANDB_INIT:
    wandb.login()
    wandb.init(project="cs680v3",group="swin_large_fine_tuning",name="swin_large_fine_tuning",
        config = {
        "LR_max": CONFIG.LR_MAX,
        "WEIGHT_DECAY":CONFIG.WEIGHT_DECAY,
        "train_batch" : CONFIG.TRAIN_BATCH_SIZE,
        "test_batch"  : CONFIG.TEST_BATCH_SIZE
        })

# %% [markdown]
# # Preprocessing the Tabular Data And Image Transformation

# %%
# define_transformation for the tabular data
log_tf_col = CONFIG.TRAITS_NAME
scale_feature_col =  CONFIG.EXTRA_COLOUMN + CONFIG.TRAITS_NAME
log_transform = ColumnTransformer(transformers= [
    ('log' , FunctionTransformer( np.log10  , inverse_func=sp.special.exp10, validate=False, check_inverse = True ,feature_names_out='one-to-one') , log_tf_col)
        ] , verbose_feature_names_out=False ,remainder= 'passthrough'
          )
log_transform.set_output(transform='pandas')

std_scale =  ColumnTransformer(transformers=[('scale',StandardScaler() ,scale_feature_col )  ],
                               verbose_feature_names_out=False,
                               remainder='passthrough'
                               )
std_scale.set_output(transform='pandas')
scaling_pipeline = Pipeline(steps=[   
                        ("log" , log_transform),
                        ("std_scale" , std_scale )])



# %%
# preparing the tabular data that has been given 
BASE_DIR =CONFIG.BASE_DIR
Train_DF = pd.read_csv(BASE_DIR  +  '/train.csv')  #.iloc[:-10000]
Test_DF =  pd.read_csv(BASE_DIR  +  '/test.csv')
Test_DF[log_tf_col] =1

train_df , val_df = train_test_split(Train_DF,test_size=CONFIG.TRAIN_VAL_SPLIT_SIZE,shuffle=True)
for column in CONFIG.TRAITS_NAME:
    lower_quantile = train_df[column].quantile(CONFIG.Lower_Quantile)
    upper_quantile = train_df[column].quantile(CONFIG.Upper_Quantile)
    train_df = train_df[(train_df[column] >= lower_quantile) & (train_df[column] <= upper_quantile)]



print('JPEG Files Processing:')
train_df['file_path'] = train_df['id'].apply(lambda s: f'{BASE_DIR}/train_images/{s}.jpeg')
train_df['jpeg_bytes'] = train_df['file_path'].apply(lambda fp: open(fp, 'rb').read())

val_df['file_path'] = val_df['id'].apply(lambda s: f'{BASE_DIR}/train_images/{s}.jpeg')
val_df['jpeg_bytes'] = val_df['file_path'].apply(lambda fp: open(fp, 'rb').read())

Test_DF['file_path'] = Test_DF['id'].apply(lambda s: f'{BASE_DIR}/test_images/{s}.jpeg')
Test_DF['jpeg_bytes'] = Test_DF['file_path'].apply(lambda fp: open(fp, 'rb').read())
print('JPEG Files Processing End')    


# train_tabular = train_df.drop(columns = ['id'] + CONFIG.TRAITS_NAME)
# val_tabular =    val_df.drop(columns = ['id'] + CONFIG.TRAITS_NAME)
# test_tabular = test.drop(columns = ['id'])

# scaling of the training data !

training_scaling_pipeline = Pipeline(steps=[   
                        ("log" , log_transform),
                        ("std_scale" , std_scale )])

train_tabular_scaled =  training_scaling_pipeline.fit_transform(train_df)
validation_tabular_scaled = training_scaling_pipeline.transform(val_df)

test_tabular_scaled = training_scaling_pipeline.transform(Test_DF)
Test_DF = Test_DF.drop(columns=log_tf_col)
             
                

# some hyper parameter for lr scheduler
CONFIG.N_TRAIN_SAMPLES = len(train_df)
CONFIG.N_STEPS_PER_EPOCH = (CONFIG.N_TRAIN_SAMPLES // CONFIG.TRAIN_BATCH_SIZE)
CONFIG.N_STEPS = CONFIG.N_STEPS_PER_EPOCH * CONFIG.NUM_EPOCHS + 1   



# %%
# %%
# Image PreProcessing

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
class create_dataset(Dataset):
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




train_dataset = create_dataset(
    X_jpeg_bytes = train_tabular_scaled['jpeg_bytes'].values,
    X_tabular = train_tabular_scaled[CONFIG.EXTRA_COLOUMN].values.astype(np.float32),
    y = train_tabular_scaled[CONFIG.TRAITS_NAME].values.astype(np.float32),
    transforms= TRAIN_TRANSFORMS
)

validation_dataset = create_dataset(
    X_jpeg_bytes = validation_tabular_scaled['jpeg_bytes'].values,
    X_tabular = validation_tabular_scaled[CONFIG.EXTRA_COLOUMN].values.astype(np.float32),
    y = validation_tabular_scaled[CONFIG.TRAITS_NAME].values.astype(np.float32),
    transforms= TEST_TRANSFORMS
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=CONFIG.TRAIN_BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=  0 #psutil.cpu_count(),
)

validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=CONFIG.VAL_BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    num_workers= 0 #psutil.cpu_count(),
)

test_dataset = create_dataset(
    X_jpeg_bytes = test_tabular_scaled['jpeg_bytes'].values,
    X_tabular = test_tabular_scaled[CONFIG.EXTRA_COLOUMN].values.astype(np.float32),
    y = test_tabular_scaled["id"].values,
    transforms= TEST_TRANSFORMS
    )

test_dataloader = DataLoader(
    test_dataset,
    batch_size=CONFIG.TEST_BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    num_workers= 0 #psutil.cpu_count(),
)


# %%

class ImageBackbone_swinV2(nn.Module):
    def __init__(self, backbone_name, weight_path, out_features, fixed_feature_extractor=None):
        super().__init__()
        self.out_features = out_features
        self.backbone = timm.create_model('swin_large_patch4_window7_224.ms_in22k', pretrained=True, num_classes=out_features) #remove classifier nn.Linear
        #self.backbone = backbone_.forward_head(backbone_, pre_logits=True)
        in_features = self.backbone.num_features
        
    def forward(self, x):
        x = self.backbone(x)
        return x


# %%
def initialize_image_model( model_name   , tim_num_class=0.0, fine_tuned_weight = None,fixed_feature_extractor=True):
    model_ft  = None
    if model_name == "swinV2":
        model = ImageBackbone_swinV2(model_name,weight_path=None , out_features=tim_num_class,fixed_feature_extractor=fixed_feature_extractor)
        return model

# %%
class CustomModel(nn.Module):
    def __init__(self,input_channels,out_channels, target_features_num , tim_num_class , model_name):
        super().__init__()
        self.img_backbone = initialize_image_model(model_name=model_name ,tim_num_class=tim_num_class , fine_tuned_weight = CONFIG.TIMM_FINED_TUNED_WEIGHT,fixed_feature_extractor=False)       
    def forward(self,image,x):
        output = self.img_backbone(image) # bach * (hight*col)
        return output



class BestModelSaveCallback:
    def __init__(self, save_path):
        self.save_path = save_path
        self.best_accuracy = -1

    def __call__(self, accuracy,model):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            model.to(device = "cpu")
            torch.save(model.state_dict(), self.save_path)
            model.to(device=DEVICE)
            model.eval()
            print("saving fine tuning test predictions")
            submission_df = pd.DataFrame(columns=CONFIG.TRAITS_NAME)
            for index , batch in tqdm.tqdm(enumerate(iter(test_dataloader))):
                X_img_test = batch[0] 
                X_features  = batch[1]
                test_id  = batch[2]
                #print(batch) 
                with torch.no_grad():
                    #print(X_img_test.shape, X_features.shape)
                    y_pred = model(X_img_test.to(DEVICE),X_features.to(DEVICE)).detach().cpu().numpy()  #,X_features.to(DEVICE)
                
                    pred_pd = pd.DataFrame(columns=CONFIG.EXTRA_COLOUMN + CONFIG.TRAITS_NAME)
                    pred_pd[CONFIG.EXTRA_COLOUMN] =-1
                    pred_pd[CONFIG.TRAITS_NAME] = y_pred 

                    temp1 =   scaling_pipeline['std_scale']['scale'].inverse_transform(pred_pd)
                    temp2=    pd.DataFrame(temp1, columns=CONFIG.EXTRA_COLOUMN + CONFIG.TRAITS_NAME)
                    pred_final =   scaling_pipeline['log']['log'].inverse_transform(temp2[CONFIG.TRAITS_NAME])
                    #pred_final["id"] = test_id.cpu().detach().numpy()
                    submission_df = pd.concat([submission_df, pred_final.assign(id=test_id.cpu().detach().numpy())], ignore_index=True)
            # submission_df.to_csv('submission_self_tuning.csv', index=False)
            submission_df[["id"]  + CONFIG.TRAITS_NAME ].to_csv('swin_small_fine_tuning.csv', index=False)
            print("Submit!")
            
            


# %%

model = CustomModel(input_channels = len(CONFIG.EXTRA_COLOUMN) ,out_channels =CONFIG.TABULAR_NN_OUTPUT, target_features_num= len(CONFIG.TRAITS_NAME), tim_num_class=CONFIG.TIM_NUM_CLASS , model_name=CONFIG.TIM_MODEL_NAME)
model.to(DEVICE)



# %%

# %%
# optimizer
import torcheval.metrics
import torcheval.metrics.regression
import torchmetrics

optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=CONFIG.LR_MAX,
    weight_decay=CONFIG.WEIGHT_DECAY,
)

# lr scheduler
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
    #return torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
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
        
MSE = torcheval.metrics.regression.MeanSquaredError().to(DEVICE)
R2sc = torchmetrics.regression.R2Score(num_outputs=len(CONFIG.TRAITS_NAME) , multioutput="uniform_average").to(DEVICE)
R2sc_val = torchmetrics.regression.R2Score(num_outputs=len(CONFIG.TRAITS_NAME) , multioutput="uniform_average").to(DEVICE)
MSE_val = torcheval.metrics.regression.MeanSquaredError().to(DEVICE)
LOSS = AverageMeter()
LOSS_val = AverageMeter()
TARGET_TRAITS_MEAN = torch.tensor(train_df[CONFIG.TRAITS_NAME].mean().values).to(DEVICE)        # target mean accross the training dataset
EPS = torch.tensor([1e-6]).to('cuda')


# just to check if r2 loss is also decreasing 
def r2_loss(y_pred, y_true):
    ss_res = torch.sum((y_true - y_pred)**2, dim=0)
    ss_total = torch.sum((y_true - TARGET_TRAITS_MEAN)**2, dim=0)
    ss_total = torch.maximum(ss_total, EPS)
    r2 = torch.mean(ss_res / ss_total)
    return r2

LOSS_FN = nn.SmoothL1Loss() # r2_loss
LR_SCHEDULER = get_lr_scheduler(optimizer)


def train_batch(inputs,model):
    model.train()  
    #X_image, X_tabular, y_true
    if CONFIG.INCLUDE_EXTRA_FEATURES :
        x,z,y = inputs
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        z = z.to(DEVICE)
        y_pred = model(x,z)   
    else:
        x,y = inputs
        x = x.to(DEVICE)
        y = y.to(DEVICE)    
        y_pred = model(x)
    
    
    #loss_func = nn.MSELoss()
    #loss_val = loss_func(prediction,y)
    loss = LOSS_FN(y_pred,y) 
    LOSS.update(loss)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    LR_SCHEDULER.step()
    MSE.update(y_pred,y)
    R2sc.update(y_pred , y )
    return LOSS.avg.detach().cpu().numpy() ,MSE.compute().item() ,R2sc.compute().item()

@torch.no_grad
def do_prediction(inputs,model, is_val=False):
    global Train_std_tensor , Train_mean_tensor
    model.eval()
    if  CONFIG.INCLUDE_EXTRA_FEATURES:
        x,z,y = inputs
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        z = z.to(DEVICE)
        prediction = model(x,z)
    else:
        x,y = inputs
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        prediction = model(x)
    if is_val :
        prediction =prediction


@torch.no_grad()
def validation_loss_batch(inputs,model):
    global Train_std_tensor , Train_mean_tensor
    model.eval()
    if CONFIG.INCLUDE_EXTRA_FEATURES:
        x,z,y = inputs
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        z = z.to(DEVICE)
        prediction = model(x,z)
    else:
        x,y = inputs
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        prediction = model(x)
    prediction =prediction
    loss = LOSS_FN(prediction, y)
    LOSS_val.update(loss)
    MSE_val.update(prediction,y)
    R2sc_val.update(prediction , y )
    return LOSS_val.avg.detach().cpu().numpy() ,MSE_val.compute().item() ,R2sc_val.compute().item()

def utils_convert_to_2d_tensors(predictions,targets):
    predictions  = np.array(predictions)
    targets = np.array(targets)
    predictions  = np.reshape(predictions , (-1, predictions.shape[-1]))
    targets  = np.reshape(targets  , (-1 , targets.shape[-1]))
    return torch.Tensor(predictions), torch.Tensor(targets)


MODEL_NAME_SAVE = 'swin_large_fine_tuning_train.pth'
best_model_callback_train = BestModelSaveCallback(save_path=os.path.join(CONFIG.BASE_DIR,MODEL_NAME_SAVE))
def train(trainLoader,valLoader,model,num_epochs,best_model_callback):
    #wandb.watch(model,loss_function,log = "all",log_freq=50)
    
    train_epoch_loss , train_epoch_r2 , train_epoch_mse =[] , [] , []
    val_epoch_loss , val_epoch_r2 , val_epoch_mse = [],[],[]
    
    
    for epoch in range(num_epochs):
        if CONFIG.WANDB_INIT:
            wandb.log({"epoch":epoch })
        # print(f"epoch: {epoch} , lr is { LR_SCHEDULER.get_last_lr()}" )
        train_loss_current_epoch ,train_mse_current_epoch, train_r2_current_epoch = [] , [] , []
        MSE.reset()
        R2sc.reset()
        LOSS.reset()
        # batch training loss
        train_r2 =  -9
        with tqdm.tqdm(total=len(trainLoader)) as trainingLoop:
            for index,batch in enumerate(iter(trainLoader)): 
        
                loss,mse_ , train_r2 = train_batch(batch,model)
                # train_loss_current_epoch.append(loss)
                # train_mse_current_epoch.append(mse_)
                # train_r2_current_epoch.append(r2_)
                
                # trainingLoop.set_description(f"Epoch:{epoch} , Batch: {index}/{len(trainLoader)} ")
                # #print(loss , mse_,  r2_ ,LR_SCHEDULER.get_last_lr())
                # trainingLoop.set_postfix({ "Training batch" : index , "loss is" : loss , "MSE" :  mse_ , "R2":  r2_, "lr was":  LR_SCHEDULER.get_last_lr()[0] })
                # trainingLoop.update(1)
                if CONFIG.WANDB_INIT:
                    wandb.log({"Training-Loss":loss  , "Training-MSE" :  mse_ , "Training-R2":  train_r2 })
        
        # train_epoch_loss.append(np.array(train_loss_current_epoch).mean() )
        # train_epoch_r2.append(np.array(train_r2_current_epoch).mean())
        # train_epoch_mse.append(np.array(train_mse_current_epoch).mean())

        MSE_val.reset()
        R2sc_val.reset()
        LOSS_val.reset()
                
        # validation set loss & accuracy  
        # val_loss_current_epoch ,val_mse_current_epoch, val_r2_current_epoch = [] , [] , []
        val_r2 = 0
        with tqdm.tqdm(total = len(valLoader)) as validationLoop:
            for index,batch in enumerate(iter(valLoader)):
                loss,mse_ , val_r2 = validation_loss_batch(batch,model)
                #val_loss_current_epoch.append(loss)
                #val_mse_current_epoch.append(mse_) 
                #val_r2_current_epoch.append(r2_)
                #validationLoop.set_description(f"Batch: {index}/{len(valLoader)}")
                #validationLoop.set_postfix({ "Validation batch" : index , "loss is" : loss , "MSE" :  mse_ , "R2":  val_r2 })
                ##wandb.log({"Vlaidation loss" : loss})
                #validationLoop.update(1)
            if CONFIG.WANDB_INIT :
                wandb.log({"Validation-Loss":loss  , "Validation-MSE" :  mse_ , "Validation-R2":  val_r2 })
        
        best_model_callback(val_r2,model)        # save the best model according to the validation accuracy
        best_model_callback_train(train_r2,model)
        
    return train_epoch_loss,val_epoch_loss,train_epoch_r2 , val_epoch_r2






# %%
MODEL_NAME_SAVE = 'swin_large_fine_tuning_Val.pth'
best_model_callback = BestModelSaveCallback(save_path=os.path.join(CONFIG.BASE_DIR,MODEL_NAME_SAVE))
train_losses, val_losses , train_accuracies,val_accuracies = train(train_dataloader,validation_dataloader,model,CONFIG.NUM_EPOCHS,best_model_callback)

# %%

# swinV2 = torch.load('/home/prajwal/cs680/cs680_kaggle/data/swin_v2_fine_self_tuning2.pth')
# model.load_state_dict(swinV2)
# model.to(DEVICE)
model.eval()
submission_df = pd.DataFrame(columns=CONFIG.TRAITS_NAME)
for index , batch in tqdm.tqdm(enumerate(iter(test_dataloader))):
    X_img_test = batch[0] 
    X_features  = batch[1]
    test_id  = batch[2]
    #print(batch) 
    with torch.no_grad():
        #print(X_img_test.shape, X_features.shape)
        y_pred = model(X_img_test.to(DEVICE),X_features.to(DEVICE)).detach().cpu().numpy()  #,X_features.to(DEVICE)
    
        pred_pd = pd.DataFrame(columns=CONFIG.EXTRA_COLOUMN + CONFIG.TRAITS_NAME)
        pred_pd[CONFIG.EXTRA_COLOUMN] =-1
        pred_pd[CONFIG.TRAITS_NAME] = y_pred 

        temp1 =   scaling_pipeline['std_scale']['scale'].inverse_transform(pred_pd)
        temp2=    pd.DataFrame(temp1, columns=CONFIG.EXTRA_COLOUMN + CONFIG.TRAITS_NAME)
        pred_final =   scaling_pipeline['log']['log'].inverse_transform(temp2[CONFIG.TRAITS_NAME])
        #pred_final["id"] = test_id.cpu().detach().numpy()
        submission_df = pd.concat([submission_df, pred_final.assign(id=test_id.cpu().detach().numpy())], ignore_index=True)
# submission_df.to_csv('submission_self_tuning.csv', index=False)
submission_df[["id"]  + CONFIG.TRAITS_NAME ].to_csv('swin_small_fine_tuning.csv', index=False)
print("Submit!")



