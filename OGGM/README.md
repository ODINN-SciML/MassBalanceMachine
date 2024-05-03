This folder is purely dedicated to OGGM, as running the code on a Windows machine requires a remote environment. Future plans are to have this code with the other scripts in the data processing pipeline and automate this process with a Docker Image (and possibly with Kubernetes). If you haven't already, please consult [How to install Linux on Windows with WSL](https://learn.microsoft.com/en-us/windows/wsl/install). A list of steps is provided for Windows users to run this code on their local machine in a remote environment: 

1. Please see one of the following links, depending on your editor of choice, how to connect WSL as a remote environment:
   1. [Visual Studio](https://code.visualstudio.com/docs/remote/wsl)
   2. [PyCharm](https://www.jetbrains.com/help/pycharm/using-wsl-as-a-remote-interpreter.html#create-wsl-interpreter)
   3. [Juypyter Notebook](https://matinnuhamunada.github.io/posts/2021/04/jupyter-wsl2/)

2. Installing Anaconda on Linux: 
   1. [Anaconda Docs](https://docs.anaconda.com/free/anaconda/install/linux/), or
   2. [Steps to Install Anaconda on Windows Ubuntu Terminal](https://docs.anaconda.com/free/anaconda/install/linux/)
3. [Create a new Anaconda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) from the ```oggm_env.yml```
4. Access the remote environment, and ```conda activate oggm_yml```, and start ```jupyter notebook```.



When you run the code for the first time, the RGI directories for your RGI IDs are downloaded to the ```Data``` folder this can take some while.  