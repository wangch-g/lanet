from PIL import Image
from torch.utils.data import Dataset, DataLoader

from augmentations import ha_augment_sample, resize_sample, spatial_augment_sample
from utils import to_tensor_sample

def image_transforms(shape, jittering):
    def train_transforms(sample):
        sample = resize_sample(sample, image_shape=shape)
        sample = spatial_augment_sample(sample)
        sample = to_tensor_sample(sample)
        sample = ha_augment_sample(sample, jitter_paramters=jittering)
        return sample

    return {'train': train_transforms}

class GetData(Dataset):
    def __init__(self, config, transforms=None):
        """
        Get the list containing all images and labels.
        """
        datafile = open(config.train_txt, 'r')
        lines = datafile.readlines()

        dataset = []
        for line in lines:
            line = line.rstrip()
            data = line.split()
            dataset.append(data[0])

        self.config = config
        self.dataset = dataset
        self.root = config.train_root
        
        self.transforms = transforms
	
    def __getitem__(self, index):
        """
        Return image'data and its label.
        """
        img_path = self.dataset[index]
        img_file = self.root + img_path
        img = Image.open(img_file)
        
        # image.mode == 'L' means the image is in gray scale 
        if img.mode == 'L':
            img_new = Image.new("RGB", img.size)
            img_new.paste(img)
            sample = {'image': img_new, 'idx': index}
        else:
            sample = {'image': img, 'idx': index}

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        """
        Return the number of all data.
        """
        return len(self.dataset)

def get_data_loader(
                config,
                transforms=None,
                sampler=None,
                drop_last=True,
                ):
    """
    Return batch data for training.
    """
    transforms = image_transforms(shape=config.image_shape, jittering=config.jittering)
    dataset = GetData(config, transforms=transforms['train'])

    train_loader = DataLoader(
                        dataset,
                        batch_size=config.batch_size,
                        shuffle=config.shuffle,
                        sampler=sampler,
                        num_workers=config.num_workers,
                        pin_memory=config.pin_memory,
                        drop_last=drop_last
                        )

    return train_loader
