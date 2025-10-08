
import numpy as np
import os
import yaml
from tqdm import tqdm
import os.path as osp
from phc.utils import torch_utils
from phc.utils.isaacgym_torch_utils import quat_rotate
import joblib
import torch
import torch.multiprocessing as mp
import copy
import gc
from typing import List, Optional, Sequence, Tuple, Union
from scipy.spatial.transform import Rotation as sRot
import random
from phc.utils.flags import flags
from phc.utils.motion_lib_base import MotionLibBase, DeviceCache, compute_motion_dof_vels, FixHeightMode
from phc.utils.torch_humanoid_batch import Humanoid_Batch
from easydict import EasyDict
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)


class MotionLib():

    def __init__(self):
        self._device = 'cuda'
        self._load_motions('/mnt/Exp_HDD/dataset/test/new_data_0.015')
        # self.motion_ids = torch.arange(len(self._motions), dtype=torch.long, device=self._device)

        return

    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion(self, motion_id):
        return self._motions[motion_id]

    def get_motion_length(self, motion_ids=None):
        if motion_ids is None:
            return self._motion_lengths
        else:
            return self._motion_lengths[motion_ids]

    def sample_motions(self, n):
        m = self.num_motions
        # motion_ids = np.random.choice(m, size=n, replace=True)
        motion_ids = torch.randint(0, self.num_motions, (n,), device=self._device)

        return motion_ids

    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert (truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time.to(self._device)

    def sample_time_interval(self, motion_ids, truncate_time=None):
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert (truncate_time >= 0.0)
            motion_len -= truncate_time
        curr_fps = 1 / 50
        motion_time = ((phase * motion_len) / curr_fps).long() * curr_fps

        return motion_time

    def _load_motions(self, motion_file):
        _motions = []
        _motion_lengths = []
        _motion_fps = []
        _motion_dt = []
        _motion_num_frames = []
        _motion_files = []

        total_len = 0.0

        motion_files, _ = fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        for f in range(num_motion_files):
            curr_file = motion_files[f]
            print("Loading {:d}/{:d} motion files: {:s}".format(f + 1, num_motion_files, curr_file))
            curr_motion = np.load(curr_file, allow_pickle=True)
            motion_fps = curr_motion['fps'].item()
            curr_dt = 1.0 / motion_fps

            num_frames = curr_motion['dof_pos'].shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

            _motion_fps.append(motion_fps)
            _motion_dt.append(curr_dt)
            _motion_num_frames.append(num_frames)

            _motions.append(curr_motion)
            _motion_lengths.append(curr_len)
            
            _motion_files.append(curr_file)

        self._motion_lengths = torch.tensor(_motion_lengths, device=self._device, dtype=torch.float32)
        self._motion_fps = torch.tensor(_motion_fps, device=self._device, dtype=torch.float32)
        self._motion_dt = torch.tensor(_motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(_motion_num_frames, device=self._device)

        self.dof_pos = torch.cat([torch.as_tensor(m['dof_pos']) for m in _motions], dim=0).float().to(self._device)
        self.dof_vel = torch.cat([torch.as_tensor(m['dof_vel']) for m in _motions], dim=0).float().to(self._device)
        self.body_pos = torch.cat([torch.as_tensor(m['body_pos']) for m in _motions], dim=0).float().to(self._device)
        self.body_quat = torch.cat([torch.as_tensor(m['body_quat']) for m in _motions], dim=0).float().to(self._device)
        self.body_lin_vel = torch.cat([torch.as_tensor(m['body_lin_vel']) for m in _motions], dim=0).float().to(self._device)
        body_ang_vel = torch.cat([torch.as_tensor(m['body_ang_vel']) for m in _motions], dim=0).float().to(self._device)
        body_ang_vel_global = []
        for i in range(self.body_pos.shape[1]):
            body_ang_vel_global.append((quat_rotate(torch.as_tensor(self.body_quat[:,i,:]), torch.as_tensor(body_ang_vel[:,i,:]))).unsqueeze(1))
        self.body_ang_vel = torch.cat(body_ang_vel_global, dim=1)
        assert self.body_ang_vel.shape[-2] == 38
        # self.body_ang_vel = torch.cat([quat_rotate(torch.as_tensor(m['body_quat']), torch.as_tensor(m['body_ang_vel'])) for m in _motions], dim=0).float().to(self._device)

        # self.body_pos[:,:,2] += 0.03

        self.num_motions = len(_motions)
        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)
        self.motion_ids = torch.arange(len(_motions), dtype=torch.long, device=self._device)

        print("Loaded {:d} motions.".format(self.num_motions))

        return

    def get_motion_state(self, motion_ids, motion_times, offset=None):

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        dof_pos0 = self.dof_pos[f0l]
        dof_pos1 = self.dof_pos[f1l]
        dof_vel0 = self.dof_vel[f0l]
        dof_vel1 = self.dof_vel[f1l]
        body_pos0 = self.body_pos[f0l]
        body_pos1 = self.body_pos[f1l]
        body_rot0 = self.body_quat[f0l]
        body_rot1 = self.body_quat[f1l]
        body_lin_vel0 = self.body_lin_vel[f0l]
        body_lin_vel1 = self.body_lin_vel[f1l]
        body_ang_vel0 = self.body_ang_vel[f0l]
        body_ang_vel1 = self.body_ang_vel[f1l]


        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        if offset is None:
            body_pos = (1.0 - blend_exp) * body_pos0 + blend_exp * body_pos1  # ZL: apply offset
        else:
            body_pos = (1.0 - blend_exp) * body_pos0 + blend_exp * body_pos1 + offset[..., None, :]  # ZL: apply offset

        body_lin_vel = (1.0 - blend_exp) * body_lin_vel0 + blend_exp * body_lin_vel1
        body_ang_vel = (1.0 - blend_exp) * body_ang_vel0 + blend_exp * body_ang_vel1
        dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1
        dof_pos = (1.0 - blend) * dof_pos0 + blend * dof_pos1
        body_rot = torch_utils.slerp(body_rot0, body_rot1, blend_exp)

        return_dict = {}
    
        
        return_dict.update({
            "root_pos": body_pos[..., 0, :].clone(),
            "root_rot": body_rot[..., 0, :].clone(),
            "dof_pos": dof_pos.clone(),
            "root_vel": body_lin_vel[..., 0, :].clone(),
            "root_ang_vel": body_ang_vel[..., 0, :].clone(),
            "dof_vel": dof_vel.view(dof_vel.shape[0], -1),
            "motion_aa": None,
            "motion_bodies": None,
            "motion_limb_weights": None,
            
            "rg_pos": body_pos,
            "rb_rot": body_rot,
            "body_vel": body_lin_vel,
            "body_ang_vel": body_ang_vel,
            
            "rg_pos_t": body_pos,
            "rg_rot_t": body_rot,
            "body_vel_t": body_lin_vel,
            "body_ang_vel_t": body_ang_vel,
        })
        return return_dict

    
    def _calc_frame_blend(self, time, length, num_frames, dt):
        time = time.clone()
        phase = time / length
        phase = torch.clip(phase, 0.0, 1.0)  # clip time to be within motion length.
        time[time < 0] = 0

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0) # clip blend to be within 0 and 1
        
        return frame_idx0, frame_idx1, blend
            
        
def fetch_motion_files(motion_path: Union[str, Path], save_in_stageii: bool = True) -> Tuple[List[str], List[Optional[str]]]:
    """Find motion files under motion_path.

    motion_path may be a directory or a single file. Returns (motion_files, labels)
    where labels may contain None when no corresponding label file is found.
    """

    motion_files: List[str] = []
    labels: List[Optional[str]] = []

    directory = Path(motion_path)
    file_suffix = 'stageii.npz' if save_in_stageii else 'poses.npz'
    for file_path in directory.rglob('*' + file_suffix):
        motion_files.append(str(file_path))
        label_path = str(file_path).replace(file_suffix, 'babel.npz')
        if os.path.exists(label_path):
            labels.append(label_path)
        else:
            logger.warning("No label file found for motion file %s, using default label None", file_path)
            labels.append(None)
    logger.info("Found %d motion files in directory %s", len(motion_files), motion_path)
    return motion_files, labels


def extract_motion_according_to_labels(data: np.lib.npyio.NpzFile, splits: Optional[Sequence[Sequence[int]]]):

    dims_of_interest = ['root_pos', 'root_rot', 'dof_pos', 'dof_vel', 'root_lin_vel', 'root_ang_vel', 'projected_gravity']
    res: List[np.ndarray] = []
    length: List[int] = []
    if splits is None:
        splits = [[0, len(data['root_pos']) - 1]]

    for s in splits:
        start, end = s
        curr_res: List[np.ndarray] = []
        for d in dims_of_interest:
            curr_res.append(data[d][start:end + 1])
        curr_res = np.concatenate(curr_res, axis=-1)
        res.append(curr_res)
        length.append(end - start + 1)

    return res, length




if __name__ == "__main__":
    motion_lib = MotionLib()
    print('Finished')


    
    