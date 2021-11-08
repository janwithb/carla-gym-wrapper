# CARLA Gym Wrapper
A simple gym environment wrapping Carla, a simulator for autonomous driving research. The environment is designed for developing and comparing reinforcement learning algorithms. Trackable costs also enable the application of safe reinforcement learning algorithms.

![carla](https://user-images.githubusercontent.com/62486916/130417542-5067c3f6-57de-4295-a572-15a13493f5b5.png)

## Features
- rendering
- weather
- different observation types (state, pixel)
- traffic
- autopilot
- vehicle and map selection
- configurable environments
- collect, save and load data
- costs (for safe RL)

## Observation Space
Pixel: (3, 84, 84)

| Index         | Value             |
| ------------- |:-----------------:|
| 0             | r channel 84 x 84 |
| 1             | g channel 84 x 84 |
| 2             | b channel 84 x 84 |

State: (9, )

| Index         | Value             |
| ------------- |:-----------------:|
| 0             | x_pos             |
| 1             | y_pos             |
| 2             | z_pos             |
| 3             | pitch             |
| 4             | yaw               |
| 5             | roll              |
| 6             | acceleration      |
| 7             | angular_velocity  |
| 8             | velocity          |

## Action Space
Action: (2, )

| Index         | Value             | Min               | Max               |
| ------------- |:-----------------:|:-----------------:|:-----------------:|
| 0             | throttle_brake    | -1                | 1                 |
| 1             | steer             | -1                | 1                 |

## CARLA Setup
1. Add the following environment variables:  
```
export PYTHONPATH=$PYTHONPATH:/opt/carla-simulator/PythonAPI
export PYTHONPATH=$PYTHONPATH:/opt/carla-simulator/PythonAPI/carla/
export PYTHONPATH=$PYTHONPATH:/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg
```
2. Install the following extra libraries  
```
pip install pygame
pip install networkx
pip install dotmap
pip install gym
```

3. Open a new terminal session, and run the CARLA simulator:  
```
bash CarlaUE4.sh -fps 20
```

## Configure environments
1. Open: carla_env/\__init\__.py

2. Insert a new environment configuration
```
register(
    id='CarlaEnv-state-town01-v1',
    entry_point='carla_env.carla_env:CarlaEnv',
    max_episode_steps=500,
    kwargs={
        'render': True,
        'carla_port': 2000,
        'changing_weather_speed': 0.1,
        'frame_skip': 1,
        'observations_type': 'state',
        'traffic': True,
        'vehicle_name': 'tesla.cybertruck',
        'map_name': 'Town01',
        'autopilot': True
    }
)
```

## Environment usage
```
import gym
import carla_env


env = gym.make('CarlaEnv-pixel-v1')
env.reset()
done = False
while not done:
    action = [1, 0]
    next_obs, reward, done, info = env.step(action)
env.close()
```

Example script:
```
python carla_env_test.py
```

## Data collection
How to wrap the environment with the data collection wrapper:
```
env = gym.make('CarlaEnv-pixel-v1')
env = DataCollector(env, steps=200, save_dir='./output')
```

Load existing dataset:
```
env = DataCollector(env, steps=200, save_dir='./output', load_dir='./output/dataset_200.pkl')
```

Example script:
```
python collect_data_test.py
```

## Credits
This repository is based on code from [D4RL](https://github.com/rail-berkeley/d4rl).
