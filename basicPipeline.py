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

def createclient(x):
    return create_client(
        x, dbname=DBNAME, colname=COLLECTION, mongohost=MONGOHOST)

class RandomConvexTransform(Transform):
    def __init__(self, transforms, probs=None, **kwargs):
        super().__init__(**kwargs)
        self.transforms = transforms
        self.probs = (
            probs
            if probs is not None
            else [1 / len(transforms)] * len(transforms)
        )

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
    probs = [0.99] * 4
    transforms_list = [
        RandGaussianNoise(prob=0.1, std=0.1),
        RandBiasField(prob=0.1),
        RandAdjustContrast(prob=0.1, gamma=(0.5, 1.5)),
            # RandGaussianSmooth(
            #    prob=0.1,
            #    sigma_x=(0.5, 1.0),
            #    sigma_y=(0.5, 1.0),
            #    sigma_z=(0.5, 1.0),
            # ),
        RandGaussianSharpen(),
        RandHistogramShift(prob=0.1, num_control_points=3),
        RandGibbsNoise(prob=0.1, alpha=(0.3, 0.7)),
        RandKSpaceSpikeNoise(prob=0.1, intensity_range=(5, 13)),
        RandRicianNoise(prob=0.1, mean=0.0, std=0.1),
    ]
    transform_pipeline = RandomConvexTransform(transforms_list, probs)
    return amcollate(x, transform_pipeline, labelname=LABELNOW)



# Create a dataset
monai_dataset = MongoDataset(
    range(num_examples),
    mtransform,
    None,
    id=INDEX_ID,
    fields=VIEWFIELDS,
)

tsampler = MBatchSampler(monai_dataset, batch_size=1)
def amcollate(mlist,transform_pipeline,labelname="sublabel",inputsize=(256, 256, 256),labelsize=(256, 256, 256)):
    # def preprocess_image(img, qmin=0.01, qmax=0.99):
    #     """Quartile interval preprocessing"""
    #     img = (img - img.quantile(qmin)) / (
    #         img.quantile(qmax) - img.quantile(qmin)
    #     )
    #     return img
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

    mdict = list2dict(mlist[0])
    num_subjs = len(mdict)
    if num_subjs > 1:
        data = torch.empty(
            num_subjs, *inputsize, requires_grad=False, dtype=torch.float
        )
        labels = torch.empty(
            num_subjs, *labelsize, requires_grad=False, dtype=torch.float
        )
        for i, subj in enumerate(mdict):
            mdict[subj].sort(key=lambda _: _["id"])
            cube = np.vstack([sub["subdata"] for sub in mdict[subj]])
            orig = torch.from_numpy(cube).float()
            cube = transform_pipeline(orig)
            data[i, :] = preprocess_image(cube)
            labels[i, :] = preprocess_image(orig)
    else:
        subj = next(iter(mdict))
        mdict[subj].sort(key=lambda _: _["id"])
        cube = torch.vstack(
            [torch.from_numpy(sub["subdata"]).float() for sub in mdict[subj]]
        )
        data = transform_pipeline(cube)
        data = preprocess_image(data).unsqueeze(0)
        expected_output = preprocess_image(cube).unsqueeze(0)
        # apply a set of transformation as defined by the transformed pipeline parameter

    return data.unsqueeze(1), expected_output

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
    label = np.squeeze(y.cpu().numpy().astype(np.uint8))
    data = np.squeeze(x.cpu().numpy())
    label_nii = nib.Nifti1Image(label, img.affine, img.header)
    input_nii = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(input_nii, f"{i}_input.nii.gz")
    nib.save(label_nii, f"{i}_label.nii.gz")
    if i > 10:
        break
