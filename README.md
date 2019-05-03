# Description

Application that lets you determine what kind of poker hand is present on a photo. This is accomplished
using by detecting the position of cards and extracting their corners and processing those corners with a neural
network.

# Dependencies:

## Python >= 3.6

Anaconda distribution: https://www.anaconda.com/distribution/
Vanilla Python distribution: https://www.python.org/downloads/

Its recommended to install dependencies into a virtual Python environment:

Creating an environment with conda:
	C:> conda create -n yourname pip python=3.7

Creating an environment with virtualenv
	C:> virtualenv --system-site-packages -p python3 ./yourname

## Tensorflow >= 1.12.2

CPU Version:
	(tensorflow)C:> pip install tensorflow==1.12.2
GPU Version:
	https://www.tensorflow.org/install/gpu


## OpenCV >= 4.1.0

(yourname)C:> pip install opencv-contrib-python

## Kivy >= 1.10.1
	(yourname)C:> pip install docutils pygments pypiwin32 kivy.deps.sdl2 kivy.deps.glew
	(yourname)C:> pip install kivy

## Matplotlib >= 3.0.3
	(yourname)C:> pip install 

# Usage

main.py - Neural network operations
data.py - Pre-processing data
gui.py - Starts the application

python [SCRIPT_NAME] --help or python [SCRIPT_NAME] [COMMAND] --help
for information on usage
