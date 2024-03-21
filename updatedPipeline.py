from monai.transforms import Transform, Compose
from monai.transforms import (
    RandGaussianNoise,
    RandBiasField,
    ThresholdIntensity,
    RandAdjustContrast,
    SavitzkyGolaySmooth,
    MedianSmooth,
    RandGaussianSmooth,
    RandGaussianSharpen,
    RandHistogramShift,
    RandGibbsNoise,
    RandKSpaceSpikeNoise,
    RandRicianNoise,
    HistogramNormalize,
)

from mongoslabs.mongoloader import (
    create_client,
    collate_subcubes,
    mcollate,
    MBatchSampler,
    MongoDataset,
    MongoClient,
    mtransform,
)
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import nibabel as nib
import random
import matplotlib.pyplot as plt
MONGOHOST = "arctrdcn018.rs.gsu.edu"
DBNAME = "MindfulTensors"
COLLECTION = "HCP"
client = MongoClient("mongodb://" + MONGOHOST + ":27017")
LABELNOW = ["sublabel", "gwmlabel", "50label"][0]
INDEX_ID = "subject"
VIEWFIELDS = ["subdata", LABELNOW, "id", "subject"]
db = client[DBNAME]
posts = db[COLLECTION]
num_examples = int(posts.find_one(sort=[(INDEX_ID, -1)])[INDEX_ID] + 1)

#print(LABELNOW.shape)

def createclient(x):
    return create_client(
        x, dbname=DBNAME, colname=COLLECTION, mongohost=MONGOHOST)

class RandomConvexTransform(Transform):

    def __init__(self, transforms, probs=None, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, data):
        num_transforms = random.randint(1, len(self.transforms))
        weights /= weights.sum()

        # Apply randomly chosen transforms with generated weights
        transformed_data = data.clone()  # Initialize with the original data
        for transform, weight in zip(chosen_transforms, weights):
            transformed_data += weight * transform(data)

        return transformed_data

    def __call__(self, data):
        # Generate a random number of transforms to combine
        num_transforms = random.randint(1, len(self.transforms))
        chosen_transforms = random.sample(self.transforms, num_transforms)

        # Generate random weights that sum up to 1 for convex combination
        weights = np.random.rand(num_transforms)
        weights /= weights.sum()

        # Apply randomly chosen transforms with generated weights
        # Initialize an accumulator for the transformed data
        transformed_data = torch.zeros_like(data)
        for transform, weight in zip(chosen_transforms, weights):
            transformed = transform(
                data
            )  # Assuming transform directly works on tensor
            transformed_data += weight * transformed

        return transformed_data

def mycollate_full(x):
    return amcollate(x, labelname=LABELNOW)

# Create a dataset
monai_dataset = MongoDataset(
    range(num_examples),
    mtransform,
    None,
    id=INDEX_ID,
    fields=VIEWFIELDS,
)

tsampler = MBatchSampler(monai_dataset, batch_size=1)
def amcollate(mlist,labelname="sublabel",inputsize=(256, 256, 256),labelsize=(256, 256, 256)):
    def preprocess_image(img):
        """Unit interval preprocessing"""
        img = (img - img.min()) / (img.max() - img.min())
        return img

    def list2dict(mlist):
        mdict = {}
        for element in mlist:
            if element["subject"] in mdict:
                mdict[element["subject"]].append(element)
            else:
                mdict[element["subject"]] = [element]
        return mdict

    std_range = np.arange(0.01, 0.51, 0.05)  # Range of std values to try

    for std in std_range:
        transform_pipeline = RandGaussianNoise(prob=1.0, std=std)

        mdict = list2dict(mlist[0])
        num_subjs = len(mdict)
        data = torch.empty(num_subjs, *inputsize, requires_grad=False, dtype=torch.float)
        labels = torch.empty(num_subjs, *labelsize, requires_grad=False, dtype=torch.float)

        for i, subj in enumerate(mdict):
            mdict[subj].sort(key=lambda _: _["id"])
            cube = np.vstack([sub["subdata"] for sub in mdict[subj]])
            orig = torch.from_numpy(cube).float()
            print(f"Original Tensor: {orig.shape}")

            cube = transform_pipeline(orig)
            print(f"Transformed Tensor: {cube.shape}")
            print(torch.any(cube != orig))

            data[i, :] = orig
            labels[i, :] = cube

    return data, labels

# Create a DataLoader
tdataloader = DataLoader(
    monai_dataset,
    sampler=tsampler,
    collate_fn=mycollate_full,
    pin_memory=True,
    worker_init_fn=createclient,
    persistent_workers=True,
    prefetch_factor=3,
    num_workers=1,
)
img = nib.load("./t1_c.nii.gz")

for i, (x,y) in tqdm(enumerate(tdataloader)):
    # Here you can use the data and target for your model training
    label = np.squeeze(y.cpu().numpy().astype(np.float32))
    data = np.squeeze(x.cpu().numpy())

    label_nii = nib.Nifti1Image(label, img.affine, img.header)
    input_nii = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(input_nii, f"{i}_input.nii.gz")
    nib.save(label_nii, f"{i}_label.nii.gz")

    if i > 1:
        break
