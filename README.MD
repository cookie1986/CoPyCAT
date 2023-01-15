# CoPyCAT - A python package for measuring verbal mirroring in dyadic conversations

## Setup and installation 
Copy the directory url and enter ```git clone [URL]``` into a command prompt.

```cd``` into the root directory


### Installation via pip and virtualenv
Next, create a virtual environment via ```python3 -m venv venv```. Activate by entering ```source venv/bin/activate``` on Mac/Linux, or ```venv/Scripts/activate.bat``` on Windows.

Once inside the venv, install dependencies via ```pip install -r requirements.txt```.

### Installation via Conda
Conda users can create a virtual environment by entering ```conda create -y --name venv python```.

To install package dependencies, type ```conda install --force-reinstall -y -q --name venv -c conda-forge --file requirements.txt```.

Lastly, conda environments are activated by entering ```conda activate venv```.