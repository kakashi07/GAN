import torch
import os
from torchvision.utils import save_image


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for batch in self.dl: 
            yield to_device(batch, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)



def get_default_device():
    """Pick GPU if available, else Mac's MPS else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def save_samples(index, latent_tensors, generator,sample_dir):
    os.makedirs(sample_dir, exist_ok=True)
    fake_image = generator(latent_tensors)
    fake_image = fake_image/torch.max(fake_image)
    # print(f'Saved image stats : Max : {torch.max(fake_image)} Min : {torch.min(fake_image)}')
    fake_images = torch.clip(fake_image, 0, 1)
    
    #show fake image
#     fake_images = show_image_grid(fake_image,clip_condition=True)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(fake_images.detach().cpu(), os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)

    