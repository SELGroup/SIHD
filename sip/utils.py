from sklearn.cluster import KMeans
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def process_list(old_nums, threshold):
    nums = old_nums.copy()
    result, i = [], 0
    while i < len(nums):
        if nums[i] == threshold:
            result.append(threshold)
            i += 1
        elif nums[i] < threshold:
            tmp, merged = nums[i], False
            for j in range(i + 1, len(nums)):
                if (tmp + nums[j]) > threshold:
                    result.append(tmp)
                    merged = True
                    i = j
                    break
                else:
                    tmp += nums[j]
            if not merged:
                result.append(tmp)
                merged = True
                i = len(nums)
        else:
            result.append(threshold)
            nums[i] -= threshold
    assert sum(old_nums) == sum(result)
    for rt in result:
        assert rt <= threshold
    return result

def analyze_max_value_frequency(encoding_result):
    frequency = defaultdict(int)
    
    for num_list in encoding_result.values():
        if not num_list:
            continue
        max_val = max(num_list)
        frequency[max_val] += 1
    
    sorted_freq = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
    return sorted_freq

def process_batch(observations, next_observations, N=3, M=500):
    batch_size = observations.shape[0]
    horizon = observations.shape[1]
    feature_dim = observations.shape[2]
    
    query_vectors = []
    for bid in range(batch_size):
        obs_first = observations[bid, 0]
        
        next_obs = next_observations[bid]
        non_zero_mask = np.any(next_obs != 0, axis=1)
        last_non_zero_idx = np.where(non_zero_mask)[0][-1] if np.any(non_zero_mask) else 0
        next_last = next_obs[last_non_zero_idx]
        
        query_vector = np.concatenate([obs_first, next_last])
        query_vectors.append(query_vector)
    
    query_vectors = np.array(query_vectors)
    
    sim_matrix = cosine_similarity(query_vectors)
    
    similar_indices = {}
    for bid in range(batch_size):
        sim_scores = sim_matrix[bid]
        top_N_indices = np.argsort(sim_scores)[::-1]
        top_N_indices = top_N_indices[top_N_indices != bid][:N]
        similar_indices[bid] = top_N_indices
    
    feature_dict = defaultdict(list)
    
    for bid in range(batch_size):
        feature_set = set()
        
        obs = observations[bid]
        non_zero_mask = np.any(obs != 0, axis=1)
        current_features = obs[non_zero_mask]
        for feat in current_features:
            feat_tuple = tuple(feat)
            feature_set.add(feat_tuple)
            if len(feature_set) >= M:
                break
        
        for similar_bid in similar_indices[bid]:
            if len(feature_set) >= M:
                break
            old_feature_set = set(feature_set)
            similar_obs = observations[similar_bid]
            non_zero_mask = np.any(similar_obs != 0, axis=1)
            similar_features = similar_obs[non_zero_mask]
            for feat in similar_features:
                feat_tuple = tuple(feat)
                if feat_tuple not in feature_set:
                    feature_set.add(feat_tuple)
                    if len(feature_set) >= M:
                        feature_set = set(old_feature_set)
                        break
        
        feature_dict[bid] = np.array(list(feature_set))

    return feature_dict

def get_max_non_zero_horizon(observations):
    batch_size = observations.shape[0]
    max_horizon_indices = {}
    
    for bid in range(batch_size):
        obs = observations[bid]
        
        non_zero_mask = np.any(obs != 0, axis=1)
        
        if np.any(non_zero_mask):
            max_idx = np.where(non_zero_mask)[0][-1]
        else:
            max_idx = -1
        
        max_horizon_indices[bid] = max_idx
    
    return max_horizon_indices

def save_dict(data_dict, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("successfully save to ", file_path)

def load_dict(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'rb') as f:
            print("successfully load from ", file_path)
            return pickle.load(f)
    except (pickle.PickleError, EOFError, FileNotFoundError) as e:
        print(f"Warning: Failed to load {file_path}, error: {str(e)}")
        return None

def split_on_repeated_starts(lst):
    if len(lst) < 2:
        return [lst] if lst else [], 0, 0
    
    result = []
    start = 0
    max_length = 0
    
    for i in range(1, len(lst) - 1):
        if lst[i] == lst[i - 1]:
            continue
        # if lst[i] == lst[i + 1]:
        #     current_length = i - start
        #     if current_length > max_length:
        #         max_length = current_length
        #     result.append(i - start)
        #     start = i
        current_length = i - start
        if current_length > max_length:
            max_length = current_length
        result.append(i - start)
        start = i
    
    final_length = len(lst) - start
    if final_length > max_length:
        max_length = final_length
    result.append(final_length)

    return result

def find_row_index(arr, target_row):
    matches = np.where((arr == target_row).all(axis=1))[0]
    return matches[0] if matches.size > 0 else -1

def cosine_sim_matrix(arr):
    sim_matrix = cosine_similarity(arr)
    sim_matrix = (sim_matrix + 1) / 2
    np.fill_diagonal(sim_matrix, 0)
    return sim_matrix
