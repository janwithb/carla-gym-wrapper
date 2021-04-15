import os
import pickle

from gym import Wrapper


class DataCollector(Wrapper):
    def __init__(self, env, steps=1000, save_dir='./', load_dir='./'):
        super(DataCollector, self).__init__(env)

        self.dataset = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': []
        }

        self.steps = steps
        self.directory = save_dir
        self.current_step = 0

        if load_dir is not None:
            self._load_dataset(load_dir)
            dataset_steps = len(self.dataset['observations'])
            print(len(self.dataset['observations']))
            self.steps += dataset_steps
            self.current_step += dataset_steps
            print(self.steps)
            print(self.current_step)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if self.current_step < self.steps:
            self.dataset['observations'].append(observation)
            self.dataset['actions'].append(action)
            self.dataset['rewards'].append(reward)
            self.dataset['dones'].append(done)
        elif self.current_step == self.steps:
            self._save_dataset()
        self.current_step += 1
        return observation, reward, done, info

    def close(self):
        self.env.close()
        if self.current_step < self.steps:
            self._save_dataset()

    def _save_dataset(self):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        filename = 'dataset_' + str(self.current_step) + '.pkl'
        file = open(os.path.join(self.directory, filename), "wb")
        pickle.dump(self.dataset, file)
        file.close()

    def _load_dataset(self, load_dir):
        file = open(load_dir, "rb")
        self.dataset = pickle.load(file)
