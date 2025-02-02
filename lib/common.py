import sys
import time
import numpy as np
import re

import torch
import torch.nn as nn


class RewardTracker:
    def __init__(self, writer, stop_reward, group_rewards=1):
        self.writer = writer
        self.stop_reward = stop_reward
        self.reward_buf = []
        self.steps_buf = []
        self.group_rewards = group_rewards

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        self.total_steps = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward_steps, frame, epsilon=None):
        reward, steps = reward_steps
        self.reward_buf.append(reward)
        self.steps_buf.append(steps)
        if len(self.reward_buf) < self.group_rewards:
            return False
        reward = np.mean(self.reward_buf)
        steps = np.mean(self.steps_buf)
        self.reward_buf.clear()
        self.steps_buf.clear()
        self.total_rewards.append(reward)
        self.total_steps.append(steps)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        mean_steps = np.mean(self.total_steps[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, mean steps %.2f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards)*self.group_rewards, mean_reward, mean_steps, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        self.writer.add_scalar("steps_100", mean_steps, frame)
        self.writer.add_scalar("steps", steps, frame)
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False

class lossTracker:
    def __init__(self, writer, group_losses=1):
        self.writer = writer
        self.loss_buf = []
        self.total_loss = []
        self.steps_buf = []
        self.group_losses = group_losses
        self.capacity = group_losses*10

    def loss(self, loss, frame):
        assert (isinstance(loss, np.float))
        self.loss_buf.append(loss)
        if len(self.loss_buf) < self.group_losses:
            return False
        mean_loss = np.mean(self.loss_buf)
        self.loss_buf.clear()
        self.total_loss.append(mean_loss)
        movingAverage_loss = np.mean(self.total_loss[-100:])
        if len(self.total_loss) > self.capacity:
            self.total_loss = self.total_loss[1:]

        if frame % 20000 == 0:
            print("The mean loss is %.2f and the current loss is %.2f" %(
                movingAverage_loss, mean_loss
            ))
        self.writer.add_scalar("loss_100", movingAverage_loss, frame)
        self.writer.add_scalar("loss", mean_loss, frame)

def calc_values_of_states(states, net, device="cpu"):
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_values_v = net(states_v)
        best_action_values_v = action_values_v.max(1)[0]
        mean_vals.append(best_action_values_v.mean().item())
    return np.mean(mean_vals)


def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = exp.state
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)       # the result will be masked anyway
        else:
            last_states.append(exp.last_state)
    return states, np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), last_states


def calc_loss(batch, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = states
    next_states_v = next_states
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_actions = net(next_states_v).max(1)[1]
    next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)

def find_stepidx(text, open_str, end_str):
    regex_open = re.compile(open_str)
    regex_end = re.compile(end_str)
    matches_open = regex_open.search(text)
    matches_end = regex_end.search(text)
    return np.int(text[matches_open.span()[1]:matches_end.span()[0]])

class netPreprocessor:
    def __init__(self, net, tgt_net):
        self.net = net
        self.tgt_net = tgt_net

    def train_mode(self, batch_size):
        self.net.train()
        self.net.zero_grad()
        self.net.init_hidden(batch_size)

        self.tgt_net.eval()
        self.tgt_net.init_hidden(batch_size)

    def eval_mode(self, batch_size):
        self.net.eval()
        self.net.init_hidden(batch_size)

    def populate_mode(self, batch_size):
        self.net.eval()
        self.net.init_hidden(batch_size)


def weight_visualize(net, writer):
    for name, param in net.named_parameters():
        writer.add_histogram(name, param)

def valid_result_visualize(stats=None, writer=None, step_idx=None):
    # output the mean reward to the writer
    for key, vals in stats.items():
        if (len(stats[key]) > 0) and (np.mean(vals) != 0):
            mean_value = np.mean(vals)
            std_value = np.std(vals, ddof=1)
            writer.add_scalar(key + "_val", mean_value, step_idx)
            writer.add_scalar(key + "_std_val", std_value, step_idx)
            if (key == 'order_profits') or (key == 'episode_reward'):
                writer.add_histogram(key + "dist_val", np.array(vals))
        else:
            writer.add_scalar(key + "_val", 0, step_idx)
            writer.add_scalar(key + "_std_val", 0, step_idx)
            if (key == 'order_profits') or (key == 'episode_reward'):
                writer.add_histogram(key + "_val", 0)

    # output the reward distribution to the writer