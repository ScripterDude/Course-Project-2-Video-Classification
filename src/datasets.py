from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T
import numpy as np


##tilføjet for vi ikke skal ændre hele tiden

HPC = False
#set hpc folder from jobscript.sh
if 'DATA_DIR' in os.environ:
    HPC = True

PC_DATA_PATH = 'Data/ufc101'

root_dir = os.environ['DATA_DIR'] if HPC else PC_DATA_PATH




class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir,
    split='train', 
    transform=None
):
        self.frame_paths = sorted(glob(f'{root_dir}/frames/{split}/*/*/*.jpg'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
       
    def __len__(self):
        return len(self.frame_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        video_name = os.path.basename(os.path.dirname(frame_path))
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()
        
        frame = Image.open(frame_path).convert("RGB")

        if self.transform:
            frame = self.transform(frame)
        else:
            frame = T.ToTensor()(frame)

        return frame, label


class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir, 
    split = 'train', 
    transform = None,
    stack_frames = True
):

        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames
        
        self.n_sampled_frames = 10

    def __len__(self):
        return len(self.video_paths)
    
    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        video_frames_dir = self.video_paths[idx].split('.avi')[0].replace('videos', 'frames')
        video_frames = self.load_frames(video_frames_dir)

        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [T.ToTensor()(frame) for frame in video_frames]
        
        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)


        return frames, label
    
    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)

        return frames


class FlowVideoDataset(FrameVideoDataset):
    def __init__(self, root_dir, split='train'):
        super().__init__(root_dir=root_dir, split=split)
        # self.flows_root = sorted(glob(f'{root_dir}/flows/{split}/*/*/*.npy'))
        self.flows_root = os.path.join(root_dir, 'flows', split)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        # video_name = video_path.split('/')[-1].split('.npy')[0]
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        action = os.path.basename(os.path.dirname(video_path))
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        # flows_dir = self.flows_root[idx]
        flows_dir = os.path.join(self.flows_root, action, video_name)
        flows = self.load_flows(flows_dir)

        return flows, label

    def load_flows(self, flows_dir):
        flows = []

        for i in range(1, self.n_sampled_frames):
            # flow_file = flows_dir / f'flow_{i}_{i + 1}.npy'
            flow_file = os.path.join(flows_dir, f'flow_{i}_{i + 1}.npy')
            flow = np.load(flow_file, allow_pickle=False)
            flows.append(torch.from_numpy(flow).float())

        return torch.cat(flows, dim=0)


if __name__ == '__main__':
    from torch.utils.data import DataLoader



    transform = T.Compose([T.Resize((64, 64)),T.ToTensor()])
    # frameimage_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)
    # framevideostack_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = True)
    # framevideolist_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = False)
    # flowvideolist_dataset = FlowVideoDataset(root_dir='../Data/ucf101_noleakage', split='val')

    # frameimage_loader = DataLoader(frameimage_dataset,  batch_size=8, shuffle=False)
    # framevideostack_loader = DataLoader(framevideostack_dataset,  batch_size=8, shuffle=False)
    # framevideolist_loader = DataLoader(framevideolist_dataset,  batch_size=8, shuffle=False)
    # flowvideolist_loader = DataLoader(flowvideolist_dataset,  batch_size=8, shuffle=False)

    #for frames, labels in frameimage_loader:
    #     print(frames.shape, labels.shape) # [batch, channels, height, width]

    #for video_frames, labels in framevideolist_loader:
    #     print(45*'-')
    #     for frame in video_frames: # loop through number of frames
    #         print(frame.shape, labels.shape)# [batch, channels, height, width]

    # find the flowbound for normalizing the flows
    flow_dataset = FlowVideoDataset(root_dir='../Data/ucf101_noleakage', split='train')
    flow_loader = DataLoader(flow_dataset, batch_size=8, shuffle=False)

    flow_bound = 0.0
    for flows, _ in flow_loader:
        flow_bound = max(flow_bound, flows.abs().max().item())

    print(f'Training flow bound: {flow_bound}')

    # for video_frames, labels in framevideostack_loader:
    #
    #     print(video_frames.shape, labels.shape) # [batch, channels, number of frames, height, width]
            
