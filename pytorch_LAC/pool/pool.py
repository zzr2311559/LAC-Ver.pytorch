from collections import OrderedDict, deque
import numpy as np
from copy import deepcopy

class Pool(object):

    def __init__(self, variant):

        s_dim = variant['s_dim']
        a_dim = variant['a_dim']
        d_dim = variant['d_dim']
        self.memory_capacity = variant['memory_capacity'] # 1,000,000
        store_last_n_paths = variant['store_last_n_paths'] # 一般設置的是10 也是variant裏的num_of_training_paths 其解釋爲number of training rollouts stored for analysis
        self.paths = deque(maxlen=store_last_n_paths)
        self.reset()
        if 'history_horizon' in variant.keys():
            self.history_horizon = variant['history_horizon']
        else:
            self.history_horizon = 0
        self.memory = {
            's': np.zeros([self.history_horizon+1, s_dim]),
            'a': np.zeros([self.history_horizon+1, a_dim]),
            'd': np.zeros([self.history_horizon+1, d_dim]),
            'raw_d': np.zeros([self.history_horizon+1, d_dim]),
            'r': np.zeros([self.history_horizon+1, 1]),
            'terminal': np.zeros([self.history_horizon+1, 1]),
            's_': np.zeros([self.history_horizon+1, s_dim]),


        }



        if 'finite_horizon' in variant.keys():
            if variant['finite_horizon']:
                self.memory.update({'value': np.zeros([self.history_horizon+1, 1])}),
                self.memory.update({'r_N_': np.zeros([self.history_horizon + 1, 1])}),
                self.horizon = variant['value_horizon']
        self.memory_pointer = 0
        self.min_memory_size = variant['min_memory_size']

    def reset(self):
        self.current_path = {
            's': [],
            'a': [],
            'd': [],
            'raw_d':[],
            'r': [],
            'terminal': [],
            's_': [],
        }

    def store(self, s, a, d, raw_d, r, terminal, s_):
        transition = {'s': s, 'a': a, 'd': d,'raw_d':raw_d, 'r': np.array([r]), 'terminal': np.array([terminal]), 's_': s_}
        if len(self.current_path['s']) < 1: # 小於1説明是剛開始訓練，即這是store函數在一個episode中第一次被調用
            for key in transition.keys():
                self.current_path[key] = transition[key][np.newaxis, :] # 構建了一個2D array用於存放隨後steps的state，etc
        else:
            for key in transition.keys(): # 如果大於等於1説明需要將當前transition中的數據concatenate到之前的current_path中
                self.current_path[key] = np.concatenate((self.current_path[key],transition[key][np.newaxis,:]))

        if terminal == 1.: # 如果環境終止
            if 'value' in self.memory.keys():
                r = deepcopy(self.current_path['r'])
                path_length = len(r)
                last_r = self.current_path['r'][-1, 0]
                r = np.concatenate((r, last_r*np.ones([self.horizon+1, 1])), axis=0)
                value = [] # 这里的Value和r_N_与论文中提到的实验部分有关，可能用作于GAE中提到的baseline
                r_N_ = []
                [value.append(r[i:i+self.horizon, 0].sum()) for i in range(path_length)]
                [r_N_.append(r[i + self.horizon+1, 0]) for i in range(path_length)]
                value = np.array(value)
                r_N_ = np.array(r_N_)
                self.memory['value'] = np.concatenate((self.memory['value'], value[:, np.newaxis]), axis=0) # 为了在最前面塞个0，应该是想使得和状态的个数相同
                self.memory['r_N_'] = np.concatenate((self.memory['r_N_'], r_N_[:, np.newaxis]), axis=0) # 为了在最前面塞个0，应该是想使得和状态的个数相同
            for key in self.current_path.keys():
                self.memory[key] = np.concatenate((self.memory[key], self.current_path[key]), axis=0) # 将初始状态S_0的信息全部用0表示， 最后将完整的一个episode的信息传递给self.memory
            self.paths.appendleft(self.current_path) # 保存最近10個step的數據用於分析實驗
            self.reset()
            self.memory_pointer = len(self.memory['s'])

        return self.memory_pointer

    def sample(self, batch_size):
        if self.memory_pointer < self.min_memory_size:
            return None
        else:
            indices = np.random.choice(min(self.memory_pointer, self.memory_capacity)-1-self.history_horizon, size=batch_size, replace=False) \
                      + max(1 + self.history_horizon, 1 + self.history_horizon+self.memory_pointer-self.memory_capacity)*np.ones([batch_size], np.int)
            batch = {}

            for key in self.memory.keys():
                if 's' in key:
                    sample = [self.memory[key][indices-i] for i in range(self.history_horizon + 1)]
                    sample = np.concatenate(sample, axis=1)
                    batch.update({key: sample})
                else:
                    batch.update({key: self.memory[key][indices]})
            return batch


