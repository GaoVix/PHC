
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
        self._sim_fps = 30
        print("SIM FPS:", self._sim_fps)
        self._device = 'cuda'
        self._load_motions('/mnt/Exp_HDD/dataset/test/new_data')
        self.motion_ids = torch.arange(len(self._motions), dtype=torch.long, device=self._device)

        return
    
    def num_motions(self):
        return len(self._motions)

    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion(self, motion_id):
        return self._motions[motion_id]

    def sample_motions(self, n):
        m = self.num_motions()
        motion_ids = np.random.choice(m, size=n, replace=True)

        return motion_ids
    
    def get_motion_num_steps(self, motion_ids=None):
        if motion_ids is None:
            return (self._motion_num_frames * self._sim_fps / self._motion_fps).ceil().int()
        else:
            return (self._motion_num_frames[motion_ids] * self._sim_fps / self._motion_fps).ceil().int()

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
        curr_fps = 1 / 30
        motion_time = ((phase * motion_len) / curr_fps).long() * curr_fps

        return motion_time

    
    # def load_motions(self, motion_path, random_sample=True, start_idx=0, max_len=-1, target_heading = None):
    #     # load motion load the same number of motions as there are skeletons (humanoids)
    #     # if "gts" in self.__dict__:
    #     #     del self.gts, self.grs, self.lrs, self.grvs, self.gravs, self.gavs, self.gvs, self.dvs
    #     #     del self._motion_lengths, self._motion_fps, self._motion_dt, self._motion_num_frames, self._motion_bodies, self._motion_aa
    #     #     if "gts_t" in self.__dict__:
    #     #         self.gts_t, self.grs_t, self.gvs_t
    #     #     if flags.real_traj:
    #     #         del self.q_gts, self.q_grs, self.q_gavs, self.q_gvs

    #     motions = []
    #     _motion_lengths = []
    #     _motion_fps = []
    #     _motion_dt = []
    #     _motion_num_frames = []
    #     _motion_bodies = []
    #     _motion_aa = []

    #     torch.cuda.empty_cache()
    #     gc.collect()

    #     total_len = 0.0


    #     # import ipdb; ipdb.set_trace()
    #     self._curr_motion_ids = sample_idxes
    #     self.one_hot_motions = torch.nn.functional.one_hot(self._curr_motion_ids, num_classes = self._num_unique_motions).to(self._device)  # Testing for obs_v5
    #     self.curr_motion_keys = self._motion_data_keys[sample_idxes]
    #     self._sampling_batch_prob = self._sampling_prob[self._curr_motion_ids] / self._sampling_prob[self._curr_motion_ids].sum()

    #     print("\n****************************** Current motion keys ******************************")
    #     print("Sampling motion:", sample_idxes[:30])
    #     if len(self.curr_motion_keys) < 100:
    #         print(self.curr_motion_keys)
    #     else:
    #         print(self.curr_motion_keys[:30], ".....")
    #     print("*********************************************************************************\n")

    #     motion_files, _ = fetch_motion_files(motion_path)
    #     num_motion_files = len(motion_files)
    #     for f in range(num_motion_files):
    #         curr_file = motion_files[f]
    #         print("Loading {:d}/{:d} motion files: {:s}".format(f + 1, num_motion_files, curr_file))
    #         curr_motion = np.load(curr_file, allow_pickle=True)
    #         motion_fps = curr_motion['fps']
    #         curr_dt = 1.0 / motion_fps

    #         num_frames = curr_motion['dof_pos'].shape[0]
    #         curr_len = 1.0 / motion_fps * (num_frames - 1)

    #         self._motion_fps.append(motion_fps)
    #         self._motion_dt.append(curr_dt)
    #         self._motion_num_frames.append(num_frames)

    #         self._motions.append(curr_motion)
    #         self._motion_lengths.append(curr_len)
            
    #         self._motion_files.append(curr_file)


    #     self._motion_lengths = np.array(self._motion_lengths)
    #     self._motion_fps = np.array(self._motion_fps)
    #     self._motion_dt = np.array(self._motion_dt)
    #     self._motion_num_frames = np.array(self._motion_num_frames)
        

    #     for i in tqdm(range(len(jobs) - 1)):
    #         try:
    #             res = queue.get()
    #             res_acc.update(res)
    #         except Exception as e:
    #             logging.error(f"Error in worker process {i}: {e}")

    #     for f in tqdm(range(len(res_acc))):
    #         motion_file_data, curr_motion = res_acc[f]

    #         motion_fps = curr_motion.fps
    #         curr_dt = 1.0 / motion_fps

    #         num_frames = curr_motion.global_rotation.shape[0]
    #         curr_len = 1.0 / motion_fps * (num_frames - 1)
            
            
    #         if "beta" in motion_file_data:
    #             _motion_aa.append(motion_file_data['pose_aa'].reshape(-1, self.num_joints * 3))
    #             _motion_bodies.append(curr_motion.gender_beta)
    #         else:
    #             _motion_aa.append(np.zeros((num_frames, self.num_joints * 3)))
    #             _motion_bodies.append(torch.zeros(17))

    #         _motion_fps.append(motion_fps)
    #         _motion_dt.append(curr_dt)
    #         _motion_num_frames.append(num_frames)
    #         motions.append(curr_motion)
    #         _motion_lengths.append(curr_len)
            
                
    #         del curr_motion
            
    #     self._motion_lengths = torch.tensor(_motion_lengths, device=self._device, dtype=torch.float32)
    #     self._motion_fps = torch.tensor(_motion_fps, device=self._device, dtype=torch.float32)
    #     self._motion_bodies = torch.stack(_motion_bodies).to(self._device).type(torch.float32)
    #     self._motion_aa = torch.tensor(np.concatenate(_motion_aa), device=self._device, dtype=torch.float32)

    #     self._motion_dt = torch.tensor(_motion_dt, device=self._device, dtype=torch.float32)
    #     self._motion_num_frames = torch.tensor(_motion_num_frames, device=self._device)
    #     self._motion_limb_weights = torch.tensor(np.array(limb_weights), device=self._device, dtype=torch.float32)
    #     self._num_motions = len(motions)

    #     self.gts = torch.cat([m.global_translation for m in motions], dim=0).float().to(self._device)
    #     self.grs = torch.cat([m.global_rotation for m in motions], dim=0).float().to(self._device)
    #     self.lrs = torch.cat([m.local_rotation for m in motions], dim=0).float().to(self._device)
    #     self.grvs = torch.cat([m.global_root_velocity for m in motions], dim=0).float().to(self._device)
    #     self.gravs = torch.cat([m.global_root_angular_velocity for m in motions], dim=0).float().to(self._device)
    #     self.gavs = torch.cat([m.global_angular_velocity for m in motions], dim=0).float().to(self._device)
    #     self.gvs = torch.cat([m.global_velocity for m in motions], dim=0).float().to(self._device)
    #     self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).float().to(self._device)
        
    #     if "global_translation_extend" in motions[0].__dict__:
    #         # with
    #         self.gts_t = torch.cat([m.global_translation_extend for m in motions], dim=0).float().to(self._device)
    #         self.grs_t = torch.cat([m.global_rotation_extend for m in motions], dim=0).float().to(self._device)
    #         self.gvs_t = torch.cat([m.global_velocity_extend for m in motions], dim=0).float().to(self._device)
    #         self.gavs_t = torch.cat([m.global_angular_velocity_extend for m in motions], dim=0).float().to(self._device)
        
    #     if "dof_pos" in motions[0].__dict__:
    #         self.dof_pos = torch.cat([m.dof_pos for m in motions], dim=0).float().to(self._device)


    #     lengths = self._motion_num_frames
    #     lengths_shifted = lengths.roll(1)
    #     lengths_shifted[0] = 0
    #     self.length_starts = lengths_shifted.cumsum(0)
    #     self.motion_ids = torch.arange(len(motions), dtype=torch.long, device=self._device)
    #     motion = motions[0]
    #     self.num_bodies = self.num_joints

    #     num_motions = self.num_motions()
    #     total_len = self.get_total_length()
    #     print(f"Loaded {num_motions:d} motions with a total length of {total_len:.3f}s and {self.gts.shape[0]} frames.")
    #     return motions

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

        # self.root_pos = torch.cat([torch.as_tensor(m['root_pos']) for m in _motions], dim=0).float().to(self._device)
        # self.root_rot = torch.cat([torch.as_tensor(m['root_rot']) for m in _motions], dim=0).float().to(self._device)
        self.dof_pos = torch.cat([torch.as_tensor(m['dof_pos']) for m in _motions], dim=0).float().to(self._device)
        self.dof_vel = torch.cat([torch.as_tensor(m['dof_vel']) for m in _motions], dim=0).float().to(self._device)
        # self.root_ang_vel = torch.cat([torch.as_tensor(m['root_ang_vel']) for m in _motions], dim=0).float().to(self._device)
        # self.root_lin_vel = torch.cat([torch.as_tensor(m['root_lin_vel']) for m in _motions], dim=0).float().to(self._device)
        self.body_pos = torch.cat([torch.as_tensor(m['body_pos ']) for m in _motions], dim=0).float().to(self._device)
        self.body_quat = torch.cat([torch.as_tensor(m['body_quat']) for m in _motions], dim=0).float().to(self._device)
        self.body_lin_vel = torch.cat([torch.as_tensor(m['body_lin_vel']) for m in _motions], dim=0).float().to(self._device)
        self.body_ang_vel = torch.cat([quat_rotate(torch.as_tensor(m['body_quat']), torch.as_tensor(m['body_ang_vel'])) for m in _motions], dim=0).float().to(self._device)


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

        # root_pos0 = self.root_pos[f0l]
        # root_pos1 = self.root_pos[f1l]
        # root_rot0 = self.root_rot[f0l]
        # root_rot1 = self.root_rot[f1l]
        dof_pos0 = self.dof_pos[f0l]
        dof_pos1 = self.dof_pos[f1l]
        dof_vel0 = self.dof_vel[f0l]
        dof_vel1 = self.dof_vel[f1l]
        # root_lin_vel0 = self.root_lin_vel[f0l]
        # root_lin_vel1 = self.root_lin_vel[f1l]
        # root_ang_vel0 = self.root_ang_vel[f0l]
        # root_ang_vel1 = self.root_ang_vel[f1l]
        body_pos0 = self.body_pos[f0l]
        body_pos1 = self.body_pos[f1l]
        body_rot0 = self.body_rot[f0l]
        body_rot1 = self.body_rot[f1l]
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

    
    def _calc_frame_blend(self, time, len, num_frames, dt):
        time = time.clone()
        phase = time / len
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


    
    