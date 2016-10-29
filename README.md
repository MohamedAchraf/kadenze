# <a href="https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow/info">Creative Applications of Deep Learning w/ Tensorflow</a>

[![coursecard](imgs/cadl-coursecard.png)](https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow/info)

This repository contains homework assignments for the <a href="https://www.kadenze.com/partners/kadenze-academy">Kadenze Academy</a> course on <a href="https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow/info">Creative Applications of Deep Learning w/ Tensorflow</a>.

# Overview

| | Session | Description |
| --- | --- | --- |
|Installation| **[Installation](#installation-preliminaries)** | Setting up Python/Notebook and necessary Libraries. |
|Preliminaries| **[Preliminaries with Python](session-0)** | Basics of working with Python and images. |
|1| **[Creating a Dataset/Computing with Tensorflow](session-1)** | Working with a small dataset of images.  Dataset preprocessing.  Tensorflow basics.  Sorting/organizing a dataset. |
|2| **[TBA](session-2)** | TBA. |
|3| **[TBA](session-3)** | TBA. |
|4| **[TBA](session-4)** | TBA. |
|5| **[TBA](session-5)** | TBA. |


<a name="installation-preliminaries"></a>
# Installation Preliminaries

<!-- MarkdownTOC autolink=true autoanchor=true bracket=round -->

- [What is Notebook?](#what-is-notebook)
- [Docker Toolbox](#docker-toolbox)
- [Jupyter Notebook](#jupyter-notebook)
- [Navigating to Notebook](#navigating-to-notebook)
- [Installing Python Packages](#installing-python-packages)
- [CUDA/GPU instructions](#cudagpu-instructions)
- [Testing it](#testing-it)
- [Troubleshooting](#troubleshooting)

<!-- /MarkdownTOC -->

We will be using Jupyter Notebook.  This will be necessary for submitting the homeworks and interacting with the guided session notebooks I will provide for each assignment.  Follow along this guide and we'll see how to obtain all of the necessary libraries that we'll be using.  By the end of this, you'll have installed Jupyter Notebook, NumPy, SciPy, and Matplotlib.  While many of these libraries aren't necessary for performing the Deep Learning which we'll get to in later lectures, they are incredibly useful for manipulating data on your computer, preparing data for learning, and exploring results.

<a name="what-is-notebook"></a>
## What is Notebook?

Jupyter Notebook, previously called "iPython Notebook" prior to version 4.0, is a way of interacting with Python code using a web browser.  It is a very useful instructional tool that we will be using for all of our homework assignments.  Notebooks have the file extensions "ipynb" which are abbreviations of "iPython Notebook".  Some websites such as [nbviewer.ipython.org](http://nbviewer.ipython.org) or [www.github.com](http://www.github.com) can view `.ipynb` files directly as rendered HTML.  However, these are not *interactive* versions of the notebook, meaning, they are not running the python kernel which evaluates/interacts with the code.  So the notebook is just a static version of the code contained inside of it.

In order to interact with notebook and start coding, you will need to launch Terminal (for Mac and Linux users).  For Windows users, or for anyone having any problems with the Linux/Mac instructions, please follow the next section on [Docker Toolbox](#docker-toolbox) very closely!  If you are not a Windows user, please first try skipping over the next section and use the installation instructions in [Jupyter Notebook](#jupyter-notebook) before trying Docker as this solution will be much faster than running Docker.

<a name="docker-toolbox"></a>
## Docker Toolbox

Unforunately, at the time of this writing (July 2016), there are no binaries for Tensorflow available for Windows users.  The easiest way to get up an running is to use Docker.  Docker is a way of managing a "virtual" Linux machine on your computer which will aid the creation a machine capable of running Tensorflow.  First, please download and install the Docker Toolbox:

https://www.docker.com/products/docker-toolbox

With this installed, you'll then need to run the "Docker Quickstart Terminal" which will launch a Terminal environment running on a virtual Linux machine on your computer. A virtual machine is basically an emulation of another machine. This is important because we'll use this machine to run Linux and install all of the necessary libraries for running Tensorflow.

Note, if you have trouble launching the Docker Quickstart Terminal because you have "Hyper-V", try one of the following, as suggested by Danilo Gasques:

1) [Setting up a Windows boot option to run without Hyper-V](http://www.hanselman.com/blog/SwitchEasilyBetweenVirtualBoxAndHyperVWithABCDEditBootEntryInWindows81.aspx)

2) [Running Docker on Windows with Hyper-V installed](http://jayvilalta.com/blog/2016/04/28/installing-docker-toolbox-on-windows-with-hyper-v-installed/)

Once the Docker Quickstart Terminal is launched, run the following command (ignoring the `$` sign at the beginning of each line, which just denote that each line is a terminal command that you should type out exactly and then hit ENTER afterwards):

```shell
$ cd
$ docker-machine ip
```

You should see your virtual machine's IP address as a result of the last command.  This is the location of your virtual machine.  <b>NOTE THIS IP ADDRESS</b>, as we'll need it in a second.  Now run the following command, which will download about ~530 MB containing everything we need to run tensorflow, python, and jupyter notebook (again, ignore the "$" at the beginning of the line only)!

```shell
$ cd
$ mkdir tensorflow
$ echo /$(pwd)/tensorflow
$ docker run -it -p 8888:8888 -p 6006:6006 -v /$(pwd)/tensorflow:/notebooks --name tf pkmital/tf.0.9.0-py.3.4
```

What this is doing is first creating a directory called tensorflow in the home directory, wherever that may be for your computer.  The echo command is showing you exactly where that directory is.  Then we use docker to mirror that directory on a virutal machine.  The location of that echo'ed directory is `/notebooks` on the virtual machine.

You can also try running the docker run command with any other directory. For instance:

```shell
$ docker run -it -p 8888:8888 -p 6006:6006 -v /Users/YOURUSERNAME/Desktop:/notebooks --name tf pkmital/tf.0.9.0-py.3.4
```

Which would mean that your Desktop is where you can move files around so that on the virtual machine, you can interact with them under the `/notebooks`directory.

For OSX users, if you are installing Docker because you had installation problems using Anaconda and pip, you would instead write the following command:

```shell
$ docker run -it -p 8888:8888 -p 6006:6006 -v $(pwd)/Desktop/tensorflow:/notebooks --name tf pkmital/tf.0.9.0-py.3.4
```

This command will download everything you need to run Tensorflow on your virtual machine.

When you want to start this machine, you will launch the Docker Quickstart Terminal and then write:

```shell
$ cd
$ docker start -i tf
```

You should have a new folder "tensorflow" inside your Home directory.  This directory will be empty to begin with.  Please make sure you do everything inside this directory only or else any files you make on your virtual machine WILL BE ERASED once it is shutdown!  When you clone the CADL repository, or expand the zip file downloads contents inside this directory via your Windows machine (it will be in your Home directory under a folder "tensorflow"), then you will be able to access it via your Docker instance.

<a name="jupyter-notebook"></a>
## Jupyter Notebook

### OSX/Linux

The easiest way to ensure you have Python 3.4 or higher and Jupter Notebook is to install Anaconda for Python 3.5 located here:

https://www.continuum.io/downloads

This package will install both python and the package "ipython[notebook]", along with a ton of other very useful packages such as numpy, matplotlib, scikit-learn, scikit-image, and many others.

With everything installed, restart your Terminal application (on OSX, you can use Spotlight to find the Terminal application), and then navigate to the directory containing the "ipynb", or "iPython Notebook" file, by "cd'ing" (pronounced, see-dee-ing), into that directory.  This involves typing the command: "cd some_directory".  Once inside the directory of the notebook file, you will then type: "jupyter notebook".  If this command does not work, it means you do not have notebook installed!  Try installed anaconda as above, restart your Terminal application, or manually install notebook like so (ignore the "$" signs which just denote that this is a Terminal command that you should type out exactly and then hit ENTER!):

```shell
$ pip3 install ipython[notebook]
$ jupyter notebook
```

If you run into issues that say something such as: 

```
[W 20:37:40.543 NotebookApp] Kernel not found: None
```

Then please try first running:

```shell
$ ipython3 kernel install
```

### Windows/Docker Containers

For Windows users making use of Docker, or for OSX users that had trouble w/ the pip/Anaconda install, once inside your Docker container as outlined above, you can launch notebook like so:

```shell
$ cd /notebooks
$ jupyter notebook &
```

<a name="navigating-to-notebook"></a>
## Navigating to Notebook

After running "jupyter notebook &", you should see a message similar to:

```shell
root@182bd64f27d2:~# jupyter notebook &
[I 21:15:33.647 NotebookApp] Writing notebook server cookie secret to /root/.local/share/jupyter/runtime/notebook_cookie_secret
[W 21:15:33.712 NotebookApp] WARNING: The notebook server is listening on all IP addresses and not using encryption. This is not recommended.
[W 21:15:33.713 NotebookApp] WARNING: The notebook server is listening on all IP addresses and not using authentication. This is highly insecure and not recommended.
[I 21:15:33.720 NotebookApp] Serving notebooks from local directory: /root
[I 21:15:33.721 NotebookApp] 0 active kernels
[I 21:15:33.721 NotebookApp] The IPython Notebook is running at: http://[all ip addresses on your system]:8888/
[I 21:15:33.721 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
```

Don't worry if the IP address or command prompt look different.  Note where it says: `The IPython Notebook is running at`.  If you are running Docker (Windows users), this is where we need that IP address.  For OSX/Linux users, we'll use "localhost" so don't worry about this.  Now open up Chrome/Safari/Firefox whatever browser you like, and then navigate to:

http://localhost:8888

or for Windows users:

http://ADDRESS:8888

where ADDRESS is the ip address you should have noted down before. For instance, on my machine, I would visit the website:

http://192.168.99.100:8888

This will launch the Jupyter Notebook where you will be able to interact with the homework assignments!

<a name="installing-python-packages"></a>
## Installing Python Packages

Packages are libraries or useful extensions to the standard python libraries.  In this course, we'll be using a few including Tensorflow, NumPy, MatPlotLib, SciPy, SciKit-Image, and SciKit-Learn.  Windows users will already have these libraries since the Docker container includes these.  However, if you needed to, you can install these using "pip", which is the python package manager.  OSX/Linux users should follow these steps just to be sure they have the latest versions of these packages. In Python 3.4 and higher, `pip` comes with any standard python installation.  In order to use `pip`, you'll write:

```shell
$ pip3 install some_package
```

To get the necessary libraries:

```shell
$ pip3 install "scikit-image>=0.11.3" "numpy>=1.11.0" "matplotlib>=1.5.1" "scikit-learn>=0.17"
```

This should get you all of the libraries we need for the course, EXCEPT for tensorflow.  Tensorflow is a special case, but can be `pip` installed in much the same way by pointing pip to the github repo corresponding to your OS like so.

### Ubuntu/Linux 64-bit for Python 3.4

```shell
$ pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp34-cp34m-linux_x86_64.whl
```

### Ubuntu/Linux 64-bit for Python 3.5

```shell
$ pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp35-cp35m-linux_x86_64.whl
```

### OSX for Python 3.4 or Python 3.5

```shell
$ pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.9.0-py3-none-any.whl
```

### Other Linux/OSX varieties

You can pip install Tensorflow for most OSX/Linux setups including those that are making use of NVIDIA GPUs and CUDA using one the packages listed on this link:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md#pip-installation

If you are having trouble with pip installation, try looking here first: [Common Installation Problems](https://github.com/tensorflow/tensorflow/blob/37451589519d15207448dc2d9b1c0309de15d8db/tensorflow/g3doc/get_started/os_setup.md#common-problems).  Failing that, reach out to us on the forums, or else you may want to instead run a Docker instance as outlined in the Windows instructions above: [Setting up a Docker Container](#docker-toolbox).

<a name="cudagpu-instructions"></a>
## CUDA/GPU instructions

Note that I have not provided instructions on getting setup w/ CUDA as it is beyond the scope of this course!  If you are interested in using GPU acceleration, I highly recommend using Ubuntu Linux and setting up a machine on [Nimbix](https://www.nimbix.net) or [Amazon EC2](https://aws.amazon.com/ec2/
) using the instructions here: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md#optional-install-cuda-gpus-on-linux.  If you're using Nimbix, you can skip the install process as there is already a machine pre-installed w/ Tensorflow.  Similarly, for Amazon EC2, there are many existing "images" of machines that have Tensorflow already installed.


<a name="testing-it"></a>
## Testing it

To confirm it worked, try running:

```shell
$ python3 -c 'import tensorflow as tf; print(tf.__version__)'
```

You should see 0.9.0 be printed.

<a name="troubleshooting"></a>
## Troubleshooting

### ImportError: No module named 'tensorflow'

You may have different versions of Python installed.  You can troubleshoot this by looking at the output of:

```shell
$ which python3
$ which pip3
$ python3 --version
$ pip3 --version
$ which python
$ which pip
$ python --version
$ pip --version
```

You may simply need to install tensorflow using `pip` instead of `pip3` and/or use `python` instead of `python3`, assuming they point to a version of python which is Python 3 or higher.

### AttributeError: module 'tensorflow' has no attribute '\_\_version\_\_'

You could be running python inside a directory that contains the folder "tensorflow".  Try running python inside a different directory.


### GPU-related issues

If you encounter the following when trying to run a TensorFlow program:

```python
ImportError: libcudart.so.7.0: cannot open shared object file: No such file or directory
```

Make sure you followed the GPU installation [instructions](#optional-install-cuda-gpus-on-linux).
If you built from source, and you left the Cuda or cuDNN version empty, try specifying them
explicitly.

### Protobuf library related issues

TensorFlow pip package depends on protobuf pip package version
3.0.0b2. Protobuf's pip package downloaded from [PyPI](https://pypi.python.org)
(when running `pip install protobuf`) is a Python only library, that has
Python implementations of proto serialization/deserialization which can be 10x-50x
slower than the C++ implementation. Protobuf also supports a binary extension
for the Python package that contains fast C++ based proto parsing. This
extension is not available in the standard Python only PIP package. We have
created a custom binary pip package for protobuf that contains the binary
extension. Follow these instructions to install the custom binary protobuf pip
package :

```bash
# Ubuntu/Linux 64-bit:
$ pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/protobuf-3.0.0b2.post2-cp27-none-linux_x86_64.whl

# Mac OS X:
$ pip install --upgrade https://storage.googleapis.com/tensorflow/mac/protobuf-3.0.0b2.post2-cp27-none-any.whl
```

and for Python 3 :

```bash
# Ubuntu/Linux 64-bit:
$ pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/protobuf-3.0.0b2.post2-cp34-none-linux_x86_64.whl

# Mac OS X:
$ pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/protobuf-3.0.0b2.post2-cp35-none-any.whl
```

Install the above package _after_ you have installed TensorFlow via pip, as the
standard `pip install tensorflow` would install the python only pip package. The
above pip package will over-write the existing protobuf package.
Note that the binary pip package already has support for protobuf larger than
64MB, that should fix errors such as these :

```bash
[libprotobuf ERROR google/protobuf/src/google/protobuf/io/coded_stream.cc:207] A
protocol message was rejected because it was too big (more than 67108864 bytes).
To increase the limit (or to disable these warnings), see
CodedInputStream::SetTotalBytesLimit() in google/protobuf/io/coded_stream.h.

```

### Cannot import name 'descriptor'

```python
ImportError: Traceback (most recent call last):
  File "/usr/local/lib/python3.4/dist-packages/tensorflow/core/framework/graph_pb2.py", line 6, in <module>
    from google.protobuf import descriptor as _descriptor
ImportError: cannot import name 'descriptor'
```

If you the above error when upgrading to a newer version of TensorFlow, try
uninstalling both TensorFlow and protobuf (if installed) and re-installing
TensorFlow (which will also install the correct protobuf dependency).

### Can't find setup.py

If, during `pip install`, you encounter an error like:

```bash
...
IOError: [Errno 2] No such file or directory: '/tmp/pip-o6Tpui-build/setup.py'
```

Solution: upgrade your version of pip:

```bash
pip install --upgrade pip
```

This may require `sudo`, depending on how pip is installed.

### SSLError: SSL_VERIFY_FAILED

If, during pip install from a URL, you encounter an error like:

```bash
...
SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed
```

Solution: Download the wheel manually via curl or wget, and pip install locally.

### Something Else!

Post on the [Forums](https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow-i/forums?sort=recent_activity) or check on the Tensorflow [README](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md#pip-installation)
