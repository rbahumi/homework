# CS294-112 HW 3: Q-Learning

Dependencies:
 * Python **3.5**
 * Numpy version **1.14.5**
 * TensorFlow version **1.10.5**
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**
 * seaborn
 * Box2D==**2.3.2**
 * OpenCV
 * ffmpeg

Before doing anything, first replace `gym/envs/box2d/lunar_lander.py` with the provided `lunar_lander.py` file.

The only files that you need to look at are `dqn.py` and `train_ac_f18.py`, which you will implement.

See the [HW3 PDF](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw3.pdf) for further instructions.

The starter code was based on an implementation of Q-learning for Atari generously provided by Szymon Sidor from OpenAI.

#### Setting the environment
```
# For OSX only install swig:
# brew install swig
# brew-cask install xquartz
# brew install ffmpeg

# Create a virtual environment
virtualenv -p /usr/local/bin/python3 hw3_venv

# activate and install packages
source hw3_venv/bin/activate
pip install -r requirements.txt

# Save a copy of the original lunar_lander.py
cp hw3_venv/lib/python3.7/site-packages/gym/envs/box2d/lunar_lander.py hw3_venv/lib/python3.7/site-packages/gym/envs/box2d/lunar_lander_orig.py
# Copy the new 'lunar_lander.py' file
cp lunar_lander.py hw3_venv/lib/python3.7/site-packages/gym/envs/box2d/lunar_lander.py
```
