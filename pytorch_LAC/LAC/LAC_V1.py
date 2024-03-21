import math
import time
import torch
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from copy import deepcopy
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from collections import OrderedDict, deque

from .utils import evaluate_training_rollouts
from robustness_eval import training_evaluation
from variant import *
from pool.pool import Pool
import logger
"""
注意模型还不能save参数
"""
torch.autograd.set_detect_anomaly(True)


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

LAMBDA_MIN = 0
LAMBDA_MAX = 1 

class LAC(object):
    def __init__(self,
                 a_dim,
                 s_dim,
                 variant,
                 action_space, #这里的action_space只在SAC中出现过，不在LAC中
                 action_prior='uniform',
                 ):
        self.gamma = variant['gamma']
        self.tau = variant['tau']
        self.target_update_interval = 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = variant['batch_size']
        self.approx_value = True if 'approx_value' not in variant.keys() else variant['approx_value']
        self._action_prior = action_prior
        self.s_dim = s_dim * (variant['history_horizon']+1)
        self.a_dim = a_dim
        self.history_horizon = variant['history_horizon']
        self.working_memory = deque(maxlen=variant['history_horizon']+1) # 1
        target_entropy = variant['target_entropy'] # None
        if target_entropy is None:
            self.target_entropy = -self.a_dim   #lower bound of the policy entropy # -1 for CartPole
        else:
            self.target_entropy = target_entropy
        self.finite_horizon = variant['finite_horizon'] # False
        self.soft_predict_horizon = variant['soft_predict_horizon'] # False

        labda = torch.tensor(variant['labda']) #   1. LAC 
        alpha = torch.tensor(variant['alpha']) #   2.     
        self.alpha3 = torch.tensor(variant['alpha3']) # .1 LAC 
        
        self.lyapunov = QNetwork(s_dim, a_dim, hidden_dim=64).to(device=self.device) # 这里我们实例化了一个网络，而在LAC中是直接调用了神经网络函数，因为LAC中网络是一个函数而不是类
        self.lyapunov_optim = Adam(self.lyapunov.parameters(), lr=variant['lr_l'])

        self.target_lyapunov = QNetwork(s_dim, a_dim, hidden_dim=64).to(device=self.device)
        
        hard_update(self.target_lyapunov, self.lyapunov)

        '''
        assume that automatic_entropy_tuning is True
        '''
        self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
        self.log_alpha = torch.tensor(torch.log(alpha), requires_grad=True, device=self.device)
        self.alpha = torch.exp(self.log_alpha)
        self.alpha_optim = Adam([self.log_alpha], lr=variant['lr_a'])


        self.log_labda = torch.tensor(torch.log(labda), requires_grad=True, device=self.device)
        self.labda = torch.clamp(torch.exp(self.log_labda), min=LAMBDA_MIN, max=LAMBDA_MAX)
        self.labda_optim = Adam([self.log_labda], lr=variant['lr_a'])

        self.policy = MultiGaussianPolicy(s_dim, a_dim, action_space, hidden_dim=64).to(self.device) # 这里的action_space就是env.action_space，但是由于在LAC中对环境进行了覆写，因此需要进一步研究
        self.policy_optim = Adam(self.policy.parameters(), lr=variant['lr_a'])
        
        self.target_policy = MultiGaussianPolicy(s_dim, a_dim, action_space, hidden_dim=64).to(self.device)

        hard_update(self.target_policy, self.policy)
        
        self.use_lyapunov = variant['use_lyapunov']  #True
        self.adaptive_alpha = variant['adaptive_alpha'] #True


    def learn(self, LR_A, LR_C, LR_L, LR_lag, batch, updates=0):
        S = batch['s']
        a_input = batch['a']
        S_ = batch['s_']
        R = batch['r']
        terminal = batch['terminal']
        LR_C = LR_C # 注意这里定义的优化器参数是为了以后的lr decay做准备的，但是目前并没有真正起作用，原因在于优化器是在init函数中被定义的，此时已经传入了lr参数，而后续针对lr的更新是在learn函数中进行的，并没有真正对优化器作用
        LR_A = LR_A
        LR_L = LR_L
        LR_lag = LR_lag
        
        if self.finite_horizon:
            V = batch['value']
            R_N_ = batch['r_N_']

        
        
        '''
        这一段应该是单独放入更新函数中的
        '''
        S = torch.FloatTensor(S).to(self.device)
        S_ = torch.FloatTensor(S_).to(self.device)
        a_input = torch.FloatTensor(a_input).to(self.device)
        R = torch.FloatTensor(R).to(self.device)
        terminal = torch.FloatTensor(terminal).to(self.device)
        
        if self.finite_horizon:
            V = torch.FloatTensor(V).to(self.device)
            R_N_ = torch.FloatTensor(R_N_).to(self.device)





        with torch.no_grad():
            a_, log_prob_, _ = self.target_policy.sample(S_) # !!!使用ema 注意这里的reuse=True意味着trainable=False，说明对应于SAC中的torch no grad，需要在后续的代码中注意
            l_  = self.target_lyapunov(S_, a_) # !!!使用ema 注意这里的reuse=True意味着trainable=False，说明对应于SAC中的torch no grad，需要在后续的代码中注意
            if self.approx_value:
                if self.finite_horizon:
                    if self.soft_predict_horizon:
                        l_target = R - R_N_ + l_.detach()
                    else:
                        l_target = V
                else:
                    l_target = R + self.gamma * (1-terminal) * l_.detach()
            else:
                l_target = R


        l = self.lyapunov(S, a_input) # 先放在这里，之后再放到更新函数中

        l_error = F.mse_loss(l_target, l)

        self.lyapunov_optim.zero_grad()
        l_error.backward(retain_graph=True)
        self.lyapunov_optim.step()

        a, log_prob, deterministic_a = self.policy.sample(S) # 先放在这里，之后再放到更新函数中

        lya_a_, _, _ = self.policy.sample(S_) # 注意这里的reuse=True意味着trainable=False，说明对应于SAC中的torch no grad，需要在后续的代码中注意
        selfl_  = self.lyapunov(S_, lya_a_) # 注意这里的reuse=True意味着trainable=False，说明对应于SAC中的torch no grad，需要在后续的代码中注意

        # Lyapunov constraint
        l_derta = (selfl_ - l + (self.alpha3) * R).mean()




        if self._action_prior == 'uniform':
            policy_prior_log_probs = 0.0
        
        if self.use_lyapunov is True:
            # a_loss = (self.alpha * log_prob).mean()
            a_loss = self.labda * l_derta + self.alpha * log_prob.mean() - policy_prior_log_probs # 对应于sac.py L81， 这里的a_loss实际上就是SAC中的policy_loss
        
        # print(a_loss)
        self.policy_optim.zero_grad()
        a_loss.backward(retain_graph=True)
        self.policy_optim.step()
        
        
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        labda_loss = -(self.log_labda * l_derta).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()
        alpha_tlogs = self.alpha.clone() # SAC pytorch 的写法，不知道有无冲突

        self.labda_optim.zero_grad()
        labda_loss.backward()
        self.labda_optim.step()
        self.labda = self.log_labda.exp()
        labda_tlogs = self.labda.clone() # SAC pytorch 的写法，不知道有无冲突



        if updates % self.target_update_interval == 0:
            soft_update(self.target_policy, self.policy, self.tau)
            soft_update(self.target_lyapunov, self.lyapunov, self.tau)
        

        return self.labda, self.alpha, l_error, (torch.mul(log_prob, -1)).mean(), a_loss

    

    def store_transition(self, s, a,d, r, l_r, terminal, s_):
        transition = np.hstack((s, a, d, [r], [l_r], [terminal], s_))
        index = self.pointer % self.memory_capacity  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer = self.pointer + 1


    """
    未开始transfer evaluate部分、save部分、restore部分
    """
    # def evaluate_value(self, s, a):

    #     if len(self.working_memory) < self.history_horizon:
    #         [self.working_memory.appendleft(s) for _ in range(self.history_horizon)]

    #     self.working_memory.appendleft(s)
    #     try:
    #         s = np.concatenate(self.working_memory)
    #     except ValueError:
    #         print(s)

    #     return self.sess.run(self.l, {self.S: s[np.newaxis, :], self.a_input: a[np.newaxis, :]})[0]


    def save_result(self, path=None):

        if path == None:
            raise NotADirectoryError('Save path is not specified')
        else:
            torch.save({'policy_net': self.policy.state_dict(),
                        'lyapunov_net': self.lyapunov.state_dict(),
                        'target_policy_net': self.target_policy.state_dict(),
                        'target_lyapunov_net': self.target_lyapunov.state_dict(),
                        'lyapunov_optimizer': self.lyapunov_optim.state_dict(),
                        'policy_optimizer': self.policy_optim.state_dict()}, path + '/model_params')
            
            print("Save model to path: ", path)


    def restore(self, path, evaluate=False):
        
        if path == None:
            raise NotADirectoryError('The file that is loaded from is not specified')
        else:
            print('Loading models from {}'.format(path))
            checkpoint = torch.load(path + '/')
            if checkpoint is None:
                success_load = False
                return success_load
            self.policy.load_state_dict(checkpoint['policy_net'])
            self.lyapunov.load_state_dict(checkpoint['lyapunov_net'])
            self.target_policy.load_state_dict(checkpoint['target_policy_net'])
            self.target_lyapunov.load_state_dict(checkpoint['target_lyapunov_net'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer'])
            self.lyapunov_optim.load_state_dict(checkpoint['lyapunov_optimizer'])
            success_load = True
            return success_load
            # 这里在sacpytorch算法中，给出了model.eval() & model.train()但是在LAC中似乎有别的处理方式，因此暂时不实现SAC中的方法




    def choose_action(self, state, evaluation = False):
        if len(self.working_memory) < self.history_horizon:
            [self.working_memory.appendleft(state) for _ in range(self.history_horizon)]

        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        self.working_memory.appendleft(state)
        # try:
        #     state = np.concatenate(self.working_memory)
        # except ValueError:
        #     print(state)

        if evaluation is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        
        return action.detach().cpu().numpy()[0]
                  


# 注意在TianYan中，112 和 122 有两次取最小值的操作， 对应于sac 也有两个，分别是 65 79， 122-65， 112-79， 注意到 65和122的位置均有不计算梯度的操作


def train(variant):
    env_name = variant['env_name']
    env = get_env_from_name(env_name)

    env_params = variant['env_params']

    max_episodes = env_params['max_episodes']
    max_ep_steps = env_params['max_ep_steps']
    max_global_steps = env_params['max_global_steps']
    store_last_n_paths = variant['num_of_training_paths']
    evaluation_frequency = variant['evaluation_frequency'] # 2048

    policy_params = variant['alg_params']
    policy_params['network_structure'] = env_params['network_structure']



    min_memory_size = policy_params['min_memory_size']
    steps_per_cycle = policy_params['steps_per_cycle'] # 100
    train_per_cycle = policy_params['train_per_cycle'] # 80
    batch_size = policy_params['batch_size'] # 256

    lr_a, lr_c, lr_l = policy_params['lr_a'], policy_params['lr_c'], policy_params['lr_l']
    lr_a_now = lr_a  # learning rate for actor
    lr_c_now = lr_c  # learning rate for critic
    lr_l_now = lr_l  # learning rate for critic

    if 'Fetch' in env_name or 'Hand' in env_name:
        s_dim = env.observation_space.spaces['observation'].shape[0]\
                + env.observation_space.spaces['achieved_goal'].shape[0]+ \
                env.observation_space.spaces['desired_goal'].shape[0]
    else:
        s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    # if disturber_params['process_noise']:
    #     d_dim = disturber_params['noise_dim']
    # else:
    #     d_dim = env_params['disturbance dim']

    a_upperbound = env.action_space.high
    a_lowerbound = env.action_space.low
    policy = LAC(a_dim,s_dim, policy_params, action_space=env.action_space)

    pool_params = {
        's_dim': s_dim,
        'a_dim': a_dim,
        'd_dim': 1,
        'store_last_n_paths': store_last_n_paths,
        'memory_capacity': policy_params['memory_capacity'],
        'min_memory_size': policy_params['min_memory_size'],
        'history_horizon': policy_params['history_horizon'],
        'finite_horizon':policy_params['finite_horizon']
    }
    if 'value_horizon' in policy_params.keys():
        pool_params.update({'value_horizon': policy_params['value_horizon']})
    else:
        pool_params['value_horizon'] = None
    pool = Pool(pool_params)
    # For analyse
    Render = env_params['eval_render']

    # Training setting
    starting_time = time.time()
    global_step = 0
    last_training_paths = deque(maxlen=store_last_n_paths)
    training_started = False

    log_path = variant['log_path']
    logger.configure(dir=log_path, format_strs=['csv'])
    logger.logkv('tau', policy_params['tau'])

    logger.logkv('alpha3', policy_params['alpha3'])
    logger.logkv('batch_size', policy_params['batch_size'])
    logger.logkv('target_entropy', policy.target_entropy)

    for i in range(max_episodes): # 對於每個episode

        current_path = {'rewards': [],
                        'a_loss': [],

                        'alpha': [],

                        'lambda': [],
                        'lyapunov_error': [],
                        'entropy': [],

                        }

        if global_step > max_global_steps:
            break

        s = env.reset()
        if 'Fetch' in env_name or 'Hand' in env_name:
            s = np.concatenate([s[key] for key in s.keys()])

        for j in range(max_ep_steps): # 對於每個step
            if Render:
                env.render()
            a = policy.choose_action(s)
            # a = a*0
            # action = a_lowerbound + (a + 1.) * (a_upperbound - a_lowerbound) / 2 # action rescaling

            # Run in simulator
            disturbance_input = np.zeros([a_dim + s_dim])

            s_, r, done, info = env.step(a)

            if 'Fetch' in env_name or 'Hand' in env_name:
                s_ = np.concatenate([s_[key] for key in s_.keys()])
                if info['done'] > 0:
                    done = True

            if training_started:
                global_step = global_step + 1

            if j == max_ep_steps - 1: # 如果對於一個episode，其step數量達到上限，也視爲完成，set done = True
                done = True

            terminal = 1. if done else 0. # done在達到step上限或者在環境中結束都會置爲True
            pool.store(s, a, np.zeros([1]), np.zeros([1]), r, terminal, s_)
            # policy.store_transition(s, a, disturbance, r,0, terminal, s_)

            if pool.memory_pointer > min_memory_size and global_step % steps_per_cycle == 0: # 完成了至少min_memory_size = 1000個step,才允许算法更新，并且
                training_started = True

                for _ in range(train_per_cycle):
                    batch = pool.sample(batch_size)
                    labda, alpha, l_loss, entropy, a_loss = policy.learn(lr_a_now, lr_c_now, lr_l_now, lr_a, batch) #每100個global step

            if training_started:
                current_path['rewards'].append(r)
                current_path['lyapunov_error'].append(l_loss.detach())
                current_path['alpha'].append(alpha.detach())
                current_path['lambda'].append(labda.detach())
                current_path['entropy'].append(entropy.detach())
                current_path['a_loss'].append(a_loss.detach())



            if training_started and global_step % evaluation_frequency == 0 and global_step > 0:

                logger.logkv("total_timesteps", global_step)

                training_diagnotic = evaluate_training_rollouts(last_training_paths)
                if training_diagnotic is not None:
                    if variant['num_of_evaluation_paths'] > 0:
                        eval_diagnotic = training_evaluation(variant, env, policy)
                        [logger.logkv(key, eval_diagnotic[key]) for key in eval_diagnotic.keys()]
                        training_diagnotic.pop('return')
                    [logger.logkv(key, training_diagnotic[key]) for key in training_diagnotic.keys()]
                    logger.logkv('lr_a', lr_a_now)
                    logger.logkv('lr_c', lr_c_now)
                    logger.logkv('lr_l', lr_l_now)

                    string_to_print = ['time_step:', str(global_step), '|']
                    if variant['num_of_evaluation_paths'] > 0:
                        [string_to_print.extend([key, ':', str(eval_diagnotic[key]), '|'])
                         for key in eval_diagnotic.keys()]
                    [string_to_print.extend([key, ':', str(round(training_diagnotic[key], 2)) , '|'])
                     for key in training_diagnotic.keys()]
                    print(''.join(string_to_print))

                logger.dumpkvs()
            # 状态更新
            s = s_

            # OUTPUT TRAINING INFORMATION AND LEARNING RATE DECAY
            if done:
                if training_started:
                    last_training_paths.appendleft(current_path)

                frac = 1.0 - (global_step - 1.0) / max_global_steps
                lr_a_now = lr_a * frac  # learning rate for actor
                lr_c_now = lr_c * frac  # learning rate for critic
                lr_l_now = lr_l * frac  # learning rate for critic

                break
    policy.save_result(log_path) # train 结束之后保存策略

    ending_time = time.time()
    print('Running time: ', ending_time - starting_time)
    return


























































































"""
network building
"""
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, s_dim, a_dim, hidden_dim, reuse=None):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(s_dim + a_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        # 暂时弃用Q2，LAC中不需要两个Q network
        # self.linear4 = nn.Linear(s_dim + a_dim, hidden_dim)
        # self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = self.linear1(xu)
        x1 = F.relu(x1)
        x1 = self.linear2(x1)
        x1 = F.relu(x1)
        x1 = self.linear3(x1)
        
        # 暂时弃用Q2，LAC中不需要两个Q network
        # x2 = F.relu(self.linear4(xu))
        # x2 = F.relu(self.linear5(x2))
        # x2 = self.linear6(x2)

        return x1 # , x2


class MultiGaussianPolicy(nn.Module):
    def __init__(self, s_dim, a_dim, action_space, hidden_dim, reuse=None):
        super(MultiGaussianPolicy, self).__init__()
        
        if reuse is None:
            trainable = True
        else:
            trainable = False

        self.linear1 = nn.Linear(s_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mu_linear = nn.Linear(hidden_dim, a_dim)
        self.log_sigma_linear = nn.Linear(hidden_dim, a_dim)

        self.apply(weights_init_)

        # action rescaling
        self.action_scale = torch.FloatTensor(
            (action_space.high - action_space.low) / 2.)
        self.action_bias = torch.FloatTensor(
            (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = self.linear1(state) # 增加了detach
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        mu = self.mu_linear(x)
        log_sigma = self.log_sigma_linear(x)
        log_sigma = torch.clamp(log_sigma, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mu, log_sigma

    def sample(self, state):
        mu, log_sigma = self.forward(state)
        sigma = log_sigma.exp()
        multinormal = Normal(mu, sigma)
        raw_action = multinormal.rsample()  # for reparameterization trick (mu + sigma * N(0,1))
        y_t = torch.tanh(raw_action) # 对应于clipped_a
        clipped_a = y_t * self.action_scale + self.action_bias # aciton rescaling 在LAC中没有出现
        log_prob = multinormal.log_prob(raw_action)
        # Enforcing Action Bound
        log_prob = log_prob - torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        clipped_mu = torch.tanh(mu) * self.action_scale + self.action_bias
        return clipped_a, log_prob, clipped_mu

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(MultiGaussianPolicy, self).to(device)









































"""
Soft update/ Hard update
"""
def soft_update(target, source, tau): # 后面的参数给前面
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source): # 后面的参数给前面
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)