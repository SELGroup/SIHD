from collections import namedtuple
import numpy as np
import torch
import pdb
import os

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

from sip.utils import *
from sip.sip import *
import pickle
from pathlib import Path

import math


Batch = namedtuple("Batch", "trajectories conditions")
ValueBatch = namedtuple("ValueBatch", "trajectories conditions values")


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        env="hopper-medium-replay",
        horizon=64,
        normalizer="LimitsNormalizer",
        preprocess_fns=[],
        max_path_length=1000,
        max_n_episodes=10000,
        termination_penalty=0,
        use_padding=True,
        seed=None,
        jump=1,
        jump_action=False,
        node_height=1,
    ):
        self.env_name = env
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.env.seed(seed)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        itr = sequence_dataset(env, self.preprocess_fn)
        self.jump = jump
        self.jump_action = jump_action
        self.node_height = node_height

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(
            fields, normalizer, path_lengths=fields["path_lengths"]
        )
        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        self.hierarchical_encoding()

        self.indices = self.make_indices(fields.path_lengths, horizon)

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')
    
    def hierarchical_encoding(self, tree_height=5, upper_bound=500):
        observations = self.fields["observations"]
        next_observations = self.fields["next_observations"]
        batch_size = observations.shape[0]

        feature_dict_path = "dicts/" + self.env_name + "/feature_dict.pkl"
        feature_dict = load_dict(feature_dict_path)
        if not feature_dict:
            feature_dict = process_batch(observations, next_observations, M=upper_bound)
            save_dict(feature_dict, feature_dict_path)

        encoding_dict_path = "dicts/" + self.env_name + "/encoding_dict.pkl"
        encoding_dict = load_dict(encoding_dict_path)
        if not encoding_dict:
            encoding_dict = dict()
            for bid in range(batch_size):
                print("encoding {} in range of {}".format(bid, batch_size))
                feature_arr, obs_arr = feature_dict[bid], observations[bid]
                sim_matrix = cosine_sim_matrix(feature_arr)
                etree = PartitionTree(adj_matrix=sim_matrix)
                etree.build_encoding_tree(tree_height)
                sg_list = []
                for tid in range(min(self.path_lengths[bid], upper_bound)):
                    vid = find_row_index(feature_arr, obs_arr[tid])
                    node = etree.tree_node[vid]
                    sg_list.append(node.ID)
                sg_list = split_on_repeated_starts(sg_list)

                if self.path_lengths[bid] > upper_bound:
                    last_value = sg_list[-1]
                    if last_value < self.jump:
                        if (self.path_lengths[bid] - upper_bound + last_value) <= self.jump:
                            sg_list[-1] = self.path_lengths[bid] - upper_bound + last_value
                        else:
                            sg_list[-1] = self.jump
                            sg_list.append(self.path_lengths[bid] - upper_bound + last_value - self.jump)
                    else:
                        sg_list.append(self.path_lengths[bid] - upper_bound)

                encoding_dict[bid] = sg_list
                assert sum(encoding_dict[bid]) == self.path_lengths[bid]
            save_dict(encoding_dict, encoding_dict_path)
        self.encoding_dict = encoding_dict

    def normalize(self, keys=["observations", "actions"]):
        """
        normalize fields that will be predicted by the diffusion model
        """
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes * self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f"normed_{key}"] = normed.reshape(
                self.n_episodes, self.max_path_length, -1
            )
    
    # def make_indices(self, path_lengths, horizon):
    #     """
    #     makes indices for sampling from dataset;
    #     each index maps to a datapoint
    #     """
    #     # high-level
    #     if self.jump > 1:
    #         indices, target_length = [], int(horizon / self.jump)
    #         for i, path_length in enumerate(path_lengths):
    #             sg_list = self.encoding_dict[i]
    #             sg_list = process_list(process_list(sg_list, self.jump), self.jump)
    #             max_start = min(len(sg_list) - 1, len(sg_list) + (self.max_path_length - path_length) / self.jump - horizon)
    #             if not self.use_padding:
    #                 max_start = min(max_start, len(sg_list) - target_length)
    #             for start in range(int(max_start)):
    #                 end = start + target_length
    #                 indices.append((i, int(start), int(end)))
    #             # if max_start != (len(sg_list) - 1):
    #             #     print(i, max_start, target_length, len(sg_list), path_length)
    #     # level-level
    #     else:
    #         indices = []
    #         for i, path_length in enumerate(path_lengths):
    #             max_start = min(path_length - 1, self.max_path_length - horizon)
    #             if not self.use_padding:
    #                 max_start = min(max_start, path_length - horizon)
    #             for start in range(max_start):
    #                 end = start + horizon
    #                 indices.append((i, start, end))
    #     indices = np.array(indices)
    #     return indices

    def make_indices(self, path_lengths, horizon):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint
        """
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        """
        condition on current observation for planning
        """
        return {0: observations[0][: self.observation_dim]}

    def __len__(self):
        return len(self.indices)

    # def __getitem__(self, idx, eps=1e-4):
    #     path_ind, start, end = self.indices[idx]
    #     # high-level
    #     if self.jump > 1:
    #         sg_list = self.encoding_dict[path_ind]
    #         sg_list = process_list(process_list(sg_list, self.jump), self.jump)
    #         base_index, target_index = sum(sg_list[:start]), []
    #         for i in range(start, end):
    #             assert len(target_index) == (i - start)
    #             if i < len(sg_list):
    #                 target_index.append(base_index)
    #                 base_index += sg_list[i]
    #             else:
    #                 target_index.append(base_index)
    #                 base_index += self.jump
    #         observations = self.fields.normed_observations[path_ind, target_index]

    #         if self.jump_action:
    #             actions = self.fields.normed_actions[path_ind, target_index]
    #         else:
    #             base_index, target_index = sum(sg_list[:start]), []
    #             for i in range(start, end):
    #                 assert len(target_index) == self.jump * (i - start)
    #                 if i < len(sg_list):
    #                     target_index += list(range(base_index, base_index + sg_list[i]))
    #                     base_index += sg_list[i]
    #                     target_index += [base_index - 1] * (self.jump - sg_list[i])
    #                 else:
    #                     target_index += list(range(base_index, base_index + self.jump))
    #                     base_index += self.jump
    #             actions = self.fields.normed_actions[path_ind, target_index]

    #             base_index, target_index = 0, []
    #             for i in range(start, end):
    #                 if i < len(sg_list):
    #                     target_index += list(range(base_index + sg_list[i] - 1, base_index + self.jump - 1))
    #                     base_index += self.jump
    #             actions[target_index, :] = self.fields.normed_actions[path_ind, - 1, :]
    #         actions = actions.reshape(-1, self.jump * self.action_dim)
    #         assert observations.shape[0] == actions.shape[0]
    #     # low-level
    #     else:
    #         observations = self.fields.normed_observations[path_ind, start:end][
    #             :: self.jump
    #         ]

    #         if self.jump_action:
    #             actions = self.fields.normed_actions[path_ind, start:end][:: self.jump]
    #         else:
    #             actions = self.fields.normed_actions[path_ind, start:end].reshape(
    #                 -1, self.jump * self.action_dim
    #             )
    #         assert observations.shape[0] == actions.shape[0]
        
    #     conditions = self.get_conditions(observations)
    #     if self.jump_action == "none":
    #         trajectories = observations
    #     else:
    #         trajectories = np.concatenate([actions, observations], axis=-1)

    #     batch = Batch(trajectories, conditions)
    #     return batch
    #### testing code
        # for idx in range(len(self.indices)):
        #     path_ind, start, end = self.indices[idx]
        #     observations = self.fields.normed_observations[path_ind, start:end]
        #     actions = self.fields.normed_actions[path_ind, start:end]
        #     print(path_ind, self.path_lengths[path_ind], start, end)
        #     print(observations.shape, actions.shape)

        #     old_ers, ers, sum_index = self.encoding_dict[path_ind], [], 0
        #     assert sum(old_ers) == self.path_lengths[path_ind]
        #     print(old_ers)
        #     assert sum(old_ers) > start
        #     for eid, er in enumerate(old_ers):
        #         if sum_index + er > start:
        #             if sum_index + er > end:
        #                 ers.append(end - start)
        #             else:
        #                 ers.append(sum_index + er - start)
        #                 assert (sum(old_ers[:eid + 1]) - ers[0]) == start
        #             for i in range(eid + 1, len(old_ers)):
        #                 er = old_ers[i]
        #                 if (sum(ers) + er) > (end - start):
        #                     ers.append(end - start - sum(ers))
        #                     break
        #                 else:
        #                     ers.append(er)
        #             if sum(ers) < (end - start):
        #                 ers.append(end - start - sum(ers))
        #             break
        #         else:
        #             sum_index += er
        #     print(ers, sum(ers))
        #     assert sum(ers) == (end - start)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]
        old_ers, ers, sum_index = self.encoding_dict[path_ind], [], 0
        assert sum(old_ers) == self.path_lengths[path_ind]
        assert sum(old_ers) > start
        for eid, er in enumerate(old_ers):
            if sum_index + er > start:
                if sum_index + er > end:
                    ers.append(end - start)
                else:
                    ers.append(sum_index + er - start)
                    assert (sum(old_ers[:eid + 1]) - ers[0]) == start
                for i in range(eid + 1, len(old_ers)):
                    er = old_ers[i]
                    if (sum(ers) + er) > (end - start):
                        ers.append(end - start - sum(ers))
                        break
                    else:
                        ers.append(er)
                ers = process_list(ers, self.jump)
                length = int((end - start) / self.jump)
                if len(ers) > length:
                    ers = ers[:length]
                else:
                    for i in range(len(ers), length):
                        ers.append(self.jump)
                break
            else:
                sum_index += er
        assert (self.jump * len(ers)) == (end - start)
        
        for h in range(1, self.node_height + 1):
            old_observations, old_actions, old_ers = observations.copy(), actions.copy(), ers.copy()
            observations, actions, ers = None, None, []
            if h != 1:
                old_ers = process_list(old_ers, self.jump)

            for eid in range(len(old_ers)):
                if observations is None:
                    observations = old_observations[sum(old_ers[:eid + 1])].reshape(1, -1)
                else:
                    observations = np.concatenate((observations, old_observations[sum(old_ers[:eid + 1]) - 1].reshape(1, -1)), axis=0)
                
                if self.jump_action:
                    if actions is None:
                        actions = old_actions[sum(old_ers[:eid + 1])]
                    else:
                        actions = np.concatenate((actions, old_actions[sum(old_ers[:eid + 1]) - 1].reshape(1, -1)), axis=0)
                else:
                    tmp_actions = old_actions[sum(old_ers[:eid]): sum(old_ers[:eid + 1])]
                    for _ in range(self.jump - old_ers[eid]):
                        tmp_actions = np.concatenate((tmp_actions, old_actions[sum(old_ers[:eid + 1]) - 1].reshape(1, -1)), axis=0)
                    tmp_actions = tmp_actions.reshape(-1, self.jump ** h * self.action_dim)
                    if actions is None:
                        actions = tmp_actions
                    else:
                        actions = np.concatenate((actions, tmp_actions), axis=0)

                ers.append(1)

            assert observations.shape[0] == math.ceil((end - start) / (self.jump ** h))
            assert observations.shape[1] == self.observation_dim
            assert actions.shape[0] == math.ceil((end - start) / (self.jump ** h))
            assert actions.shape[1] == (self.action_dim * (self.jump ** h))

            # while observations.shape[0] < self.jump:
            #     observations = np.concatenate((observations, observations[-1].reshape(1, -1)), axis=0)
            #     actions = np.concatenate((actions, actions[-1].reshape(1, -1)), axis=0)
            # print(observations.shape, actions.shape)

        # observations = self.fields.normed_observations[path_ind, start:end][
        #     :: self.jump
        # ]

        # if self.jump_action:
        #     actions = self.fields.normed_actions[path_ind, start:end][:: self.jump]
        # else:
        #     actions = self.fields.normed_actions[path_ind, start:end].reshape(
        #         -1, self.jump * self.action_dim
        #     )

        conditions = self.get_conditions(observations)
        if self.jump_action == "none":
            trajectories = observations
        else:
            trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch

class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        """
        condition on both the current observation and the last observation in the plan
        """
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }


class ValueDataset(SequenceDataset):
    """
    adds a value field to the datapoints for training the value function
    """
    def __init__(self, *args, discount=0.99, normed=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.normed = False
        if normed:
            self.vmin, self.vmax = self._get_bounds()
            self.normed = True

    def _get_bounds(self):
        print(
            "[ datasets/sequence ] Getting value dataset bounds...", end=" ", flush=True
        )
        vmin = np.inf
        vmax = -np.inf
        for i in range(len(self.indices)):
            value = self.__getitem__(i).values.item()
            vmin = min(value, vmin)
            vmax = max(value, vmax)
        print("✓")
        return vmin, vmax

    def normalize_value(self, value):
        ## [0, 1]
        normed = (value - self.vmin) / (self.vmax - self.vmin)
        ## [-1, 1]
        normed = normed * 2 - 1
        return normed

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields["rewards"][path_ind, start:]
        discounts = self.discounts[: len(rewards)]
        value = (discounts * rewards).sum()
        if self.normed:
            value = self.normalize_value(value)
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch

class LLValueDataset(SequenceDataset):
    """
    adds a value field to the datapoints for training the value function
    """

    def __init__(self, *args, discount=0.99, normed=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.normed = False
        if normed:
            self.vmin, self.vmax = self._get_bounds()
            self.normed = True

    def _get_bounds(self):
        print(
            "[ datasets/sequence ] Getting value dataset bounds...", end=" ", flush=True
        )
        vmin = np.inf
        vmax = -np.inf
        for i in range(len(self.indices)):
            value = self.__getitem__(i).values.item()
            vmin = min(value, vmin)
            vmax = max(value, vmax)
        print("✓")
        return vmin, vmax

    def normalize_value(self, value):
        ## [0, 1]
        normed = (value - self.vmin) / (self.vmax - self.vmin)
        ## [-1, 1]
        normed = normed * 2 - 1
        return normed

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields["rewards"][path_ind, start:]
        discounts = self.discounts[: len(rewards)]
        value = (discounts * rewards).sum()
        if self.normed:
            value = self.normalize_value(value)
        value = np.array([value], dtype=np.float32)
        # data_idx = np.where(value<0)[0]
        # if len(data_idx) > 0:
        #     print(f'negative value idx: {idx},')
        value_batch = ValueBatch(*batch, value)
        return value_batch

    def get_conditions(self, observations):
        """
        condition on current observation for planning
        """
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }
