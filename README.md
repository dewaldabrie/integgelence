INTEGGELENCE TAMAGOTCHI
=======================

Welcome to the Integgelence Tamagotchi project.

This is an attempt at a generalizable Tamagotchi zoo.

The framework supports the creation of multiple pet Tamagotchis.
The Tamagotchis run independently from each other and each has the
 following capabilities:
 * broadcasts its health and other metrics
 * can receive inputs from its
owners that affect its health.


Message passing architecture
----------------------------

Communication between the owners portal into the Tamagotchi world
and the Tamagotchis are done with message passing over TCP sockets.

This allows the Tamagotchis to be on distributed platforms.


Easy attribute extension
------------------------

The base Tamagotchi has a set of defined physical and mental attributes
which characterizes its strengths and weaknesses, likes and dislikes.

Derived classes can easily deviate from this.

The user inputs are used together with the animal attributes to change
the creatures internal physical and mental state. These states are used
to find single values for mental and physical health.

State updates are calculated in a vectorised implementation.


Easy to add new inputs
----------------------

The framework makes it easy to add new user inputs.
The Tamagotchi models can then updated to map inputs to internal state change
my adding new rows and colums to the tranformation matrices.


How to run the software
-----------------------
Install Python3.6 on your computer.
Clone onto your system from github.
Change directory into the top-level folder and then:

    $ cd tamagotchi
    $ pip install -r requirements-install.txt
    $ cd ../
    $ python run.py

How to run the tests
-----------------------
To run the tests using tox, run the following from the top-level project directory:
    $ pip install tox
    $ tox

How to run the notebook
-----------------------
Fire up the notebook to see graphs of some of the formulas used:

    $ cd notebooks
    $ jupyter notebook

Navigate to InteractionModel.ipynb

Core concepts
=============
Tamagotchi affinity
-------------------
Your creature has a set of requirements and affinities. For example:
* A large creater will require more food
* A social creature will get a mental boost from petting
* An asocial creature will be annoid by much petting

Under and over supply
---------------------
In order to improve your Tamagothi's health you can supply it with inputs such as:
* food
* petting
* injections
However, you should supply just until it reaches a healthy state. Supplying too much
indiscriminately will drive down it's health. For example, over-feeding is actually has
a bad impact on nutrition. Overpetting can similarly have a mad impact on social stimulation.


To do
-----
* do some terminal screen painting in the portal
    * https://docs.python.org/3/howto/curses.html
    * https://www.youtube.com/watch?v=eN1eZtjLEnU
* create Tamagotchi factory class with some randomness built in
* add Tamagotchi activity state (eating, sleeping, playing, dead, etc)
* add Tamagotchi energy state (activities deplete energy)
* add concept of money (owner has limited resources)
* add concept of environment that can bring cold/hot weather,
   disease outbreak, etc.
* add pet-to-pet interaction capability
* deploy a Tamagotchi to the cloud
