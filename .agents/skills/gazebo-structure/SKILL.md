---
name: gazebo-structure
description: Explains how a repository using both ROS 2 and Gazebo Should be structured. Must add the corresponding ROS 2 and Gazebo Variations to the skill call
disable-model-invocation: true
allowed-tools: Bash(git add *) Bash(git commit *) Bash(git status *) Bash(ros2) Bash(colcon colcon) Bash(ros2 launch *)
---
## Outline
- Understand the underlying transport system within Gazebo
- Understand the features and pipeline of Gazebo
- Choose a ROS 2 and Corresponding Gazebo Version
- Structure the project
- Create Gazebo System Plugins to Control Custom Simulator Behavior
- Map ROS 2/Gazebo topics and messages using a bridge
- Creating/Importing Assets to Simulate

## Gazebo Simulation Features

- Dynamics simulation: Access multiple high-performance physics engines through Gazebo Physics.

- Advanced 3D graphics: Through Gazebo Rendering, it's possible to use rendering engines such as OGRE v2 for realistic rendering of environments with high-quality lighting, shadows, and textures.

- Sensors and noise models: Generate sensor data, optionally with noise, from laser range finders, 2D/3D cameras, Kinect style sensors, contact sensors, force-torque, IMU, GPS, and more, all powered by Gazebo Sensors

- Plugins: Develop custom plugins for robot, sensor, and environment control.

- Graphical interface: Create, introspect and interact with your simulations through plugin-based graphical interfaces powered by Gazebo GUI.

- Simulation models: Access numerous robots including PR2, Pioneer2 DX, iRobot Create, and TurtleBot, and construct environments using other physically accurate models available through Gazebo Fuel. You can also build a new model using SDF.

- TCP/IP Transport: Run simulation on remote servers and interface to Gazebo Sim through socket-based message passing using Gazebo Transport.

- Command line tools: Extensive command line tools for increased simulation introspection and control.

## Setup
This project should utilize ROS 2 Version: $ARGUMENTS[0] and Gazebo Version: $ARGUMENTS[1]
Structure the project with some distinct folders:
  - $ARGUMENTS[2]_application : ROS 2 application libraries and nodes
  - $ARGUMENTS[2_]bringup: ROS 2 launch files
      - config : Configuration files (.rviz, .yaml) should live here
      - launch : All launch files should live here
  - $ARGUMENTS[2]_description : SDFormat description of simulation assets (URDF, SDF, Meshes, Materials, World SDFs)
      - hooks : Hooks styled in `dsv.in` and `sh.in` for easy compilation with CMake
      - models : SDFormat Simulation Assets
  - $ARGUMENTS[2]_gazebo : Gazebo specific system implementation
  - $ARGUMENTS[2]_examples : Folder with example implementations to be utilized
  - $ARGUMENTS[2]_msgs : Folder containing custom msg/service structures and the ability to generate them if need be.
  - docs : Folder containing markdown based documentation for the project

Every folder should contain a `CMakeLists.txt` and `package.xml` files to compile the project with ament_cmake and correctly document packages being used. Each folder should also have a small README.md to *briefly* explain that part of the package.

## Entity Component Manager (ECM)
Underneath of gz sim lies the Entity Component Manager which essentially makes every piece of simulation an entity. Each entity has one or more components attached to . For instance Entity 1 could contain a Model (with Name, Pose, Velocity, Acceleration) a Link (with Name, Pose, Velocity, Acceleration), a Visual (with Pose, Mesh URI, Material URI), a Joint (DOF, Joint Position, Joint Velocity, Joint Type), this could repeat for every entity needed.

The simulation is looped in which the server controls all the entities, and first begins by loading SDF. The loop starts with "Update Entities" > "Propagate Time" > "PreUpdate" > "Update" > "PostUpdate" > "Update Entities". Each system calls it's "Configure" function before the very first "Update Entities" call, then calls each of the corresponding functions when the server makes the main call.

## Writing Gazebo Systems
Each System (aka Plugin) in Gazebo has a certain amount of "states" which can be programmed in:
- Configure  : Called when plugin is loaded, provides ECM and SDF Attributes
- PreUpdate  : Called before update, can mutate entities and components to set forces, torques, velocity
- Update     : Called during runtime, updates physics, generally shouldn't be implemented by other systems
- PostUpdate : Called after update, cannot mutate, but can read components and publish/send events
- Reset      : Called at the end of time step, can be used to add reset-specific behavior

For further examples see the [plugin_examples](plugin_examples) directory which contains both a C++ and Python Implementation. The Python Implemenation requires extra steps

## GZ Sim Architecture
### Main Architecture
Gazebo Sim is an application entry-point of Gazebo, generally encompassing the use of all other Gazebo libraries. As an executable, it runs a simulation by launching two processes: the backend server process and frontend client process.

The core functionality of Gazebo Sim is the client-server communication and the various plugins it loads to facilitate the simulation. Plugins are libraries that are loaded at runtime. They allow Gazebo Sim to utilize other Gazebo libraries. There are many plugin types in the Gazebo framework. For example, there are plugins that introduce new physics engines or rendering engines. However, for the purpose of this document, any mention of plugins is referring only to Gazebo Sim server and GUI plugins.

Because they’re loaded at runtime, Gazebo does not need to be recompiled to add or remove plugins. Gazebo Sim ships with many plugins by default (Server plugins, Gazebo GUI plugins, Gazebo Sim GUI plugins, and more), all of which are optional and can be removed by the user. Users can also add more plugins and even write their own plugins that will be compiled into library files.

Gazebo libraries are modular. Plugins let Gazebo Sim utilize other libraries. For example, Gazebo Physics and Gazebo Sim are independent of one another, so Gazebo Sim has a physics plugin that uses Gazebo Physics as a library and incorporates Gazebo specifics, allowing Physics to be a system running in the simulation loop.

The Gazebo Physics library is only used in the physics plugin, not in other plugins nor in the core of Gazebo Sim. Some libraries are only used by one plugin, or one in each process (frontend and backend). However, some foundational libraries are used by all plugins, like Gazebo Common which provides common functionality like logging, manipulating strings, file system interaction, etc., to all plugins. Other such libraries include Gazebo Plugin, Gazebo Math, SDFormat, and more.

There are certain plugins in both the frontend and backend processes that are loaded by default in every version of Gazebo. Many other plugins are optional. One common optional plugin is the sensors plugin. Both optional and default plugins can be removed or added at any time; Gazebo Sim will continue to run with limited functionality. These demos on Server Configuration and GUI Configuration showcase that functionality.

The simulation process is depicted in the diagram below, and further explained in the Backend and Frontend process sections that follow.

### Backend server process

Gazebo Sim is responsible for loading plugins in the backend, referred to as systems. The server runs an entity-component system architecture (see Gazebo Sim terminology). The backend will usually have multiple systems responsible for everything in the simulation – computing physics, recording logs, receiving user commands, etc.

Systems act on entities and components of those entities. An entity is anything in the simulation scene (a model, link, light, actor, etc.), and components are their characteristics (pose, name, geometry, etc.). Server plugins have access to all entities and what components they have, and make modifications to those components. For example, a user could write a system that applies force to an entity by setting a component on that entity, and the physics system would pick that up and react by applying the force to make the entity actually move. The modification of entities and components is how server plugins communicate with each other.

The entity component system is self-contained in Gazebo Sim, and systems act on entities and their components. Since other Gazebo Libraries don’t have direct knowledge/access to entities and components, other Gazebo Libraries should be used in systems so that the server plugin can share entities (and their components) with these other Libraries. For example, the physics system shares entities and components with the Gazebo Physics library in order to achieve physics effects on entities and components.

There is a loop running in the backend that runs the systems, where some systems are proposing changes to entities and their components, and other systems are handling and applying those changes. This is called the “simulation loop”. The Entity Component Manager (ECM) in the backend provides the functionality for the actual querying and updating of the entities and components.

Physics, User Commands, and Scene Broadcaster are all systems launched by default in the backend. As mentioned earlier, however, even default systems can be added to or removed from the simulation loop (the Server Configuration tutorial describes how to customize default systems). For any other functionality, like sensor data processing for example, an additional system would have to be loaded. For example, if you need to generate sensor data that utilizes rendering sensors, you would need to load the sensors system. The visualization of that data, however, is left up to the client side plugins.
### Communication process

Any information that crosses the process boundary (whether back-to-front with scene broadcaster or front-to-back with user commands) has to go through Gazebo Transport, the communication library. Synchronization between processes is performed by scene broadcaster, a server-side Gazebo Sim plugin that uses the Gazebo Transport and Gazebo Messages libraries to send messages from the server to the client. The messages themselves are provided by Gazebo Messages, while the framework for creating the publishers and subscribers that exchange messages is from Gazebo Transport.

The scene broadcaster system is responsible for getting the state of the world (all of the entities and their component values) from the ever-changing simulation loop running in the back end, packaging that information into a very compact message, and periodically sending it to the frontend process. Omitting the scene broadcaster, while possible, would mean not being able to visualize data on the frontend, which can be useful for saving computational power.

In addition to state messages sent from the scene broadcaster to the client, the client can also make requests to the server using the user commands system (another backend plugin). It also utilizes the Gazebo Transport and Gazebo Messages libraries. The user commands system has services that can be requested by the frontend GUI to insert, delete, move (etc.) models, lights, and other entities, and will send responses back to the GUI acknowledging the receipt and execution of those requests.

Several non-default plugins are able to send or receive messages to other processes outside of the server-client process. For example, the server-side sensors plugin publishes messages outside of the server-client process, and the server-side diff-drive plugin receives info from processes other than the client. One such process these plugins may communicate with, for example, is ROS.
### Frontend client process

The client-side process, essentially the GUI, also comprises multiple plugins, some loaded by default, others that can be added, and all optional. All of the frontend plugins use the Gazebo GUI library. The client itself also relies on Gazebo GUI. There are visualization plugins that create the windows of the GUI, add buttons and other interactive features, and a 3D scene plugin that utilizes Gazebo Rendering that creates the scene the user sees.

Frontend plugins typically communicate among themselves using events. Events are similar to messages, but they’re processed synchronously. For example, the Render event is emitted by a 3D scene from it’s rendering thread right before the scene is rendered; this gives other plugins the chance to execute code right at that thread at that moment, which is valuable to edit the 3D scene. Other such events are emitted when the user right-clicks or hovers the scene for example.

Frontend plugins all have access to the entity and component information from the compressed message provided by the scene broadcaster system from the backend. The Entity Component Manager on the client side performs the actual message unpacking and distributes the entity and component information to the relevant plugins. This is how visualizations know how to update in accordance with backend computations. On the client side, the plugins are only reacting to the entity and component information, not interacting with it.

As mentioned in the previous section, the GUI can send commands back to the server using the server’s user commands system. This would be the case, for example, if a user wanted to insert a new model through the GUI interface. User commands receives requests from the GUI and processes them, and adds those results to entities and components in the simulation loop.

## Python Interfaces
Integrating Gazebo Sim's Python API is essential for bringing in other libraries, take a look at the [python_api](python_api) directory to see a full example
If you get permissions to compile Gazebo from source, you should modify your PYTHONPATH:

```bash
export PYTHONPATH=$PYTHONPATH:<path to ws>/install/lib/python
python3 python_api/testFixture.py
```

## Integrating ROS 2 
While the main focus is Gazebo, it is possible to integrate ROS 2 with this whole systems with two primary mechanism depending on the application, either use `ros_gz_bridge` to dynamically connect topics between ROS 2 and Gazebo. Or embed ROS 2 directly in a Gazebo system plugin. If you use `ros_gz_bridge` topics can be ROS->Gazebo, Gazebo->ROS, or ROS<->Gazebo. Topics are configured typically in a `.yaml` file.
In `ros_gz_bridge`, you get to Isolate GZ and ROS version/runtimes at the cost of limited to topics and services and limited simulator state access through transport topics. With Embedded ROS 2 nodes, you get more acces to ROS primitives and direct access to simulator state, but ROS/GZ versions are coupled.

## Credits
The ROS 2 and Gazebo Integration Best Practices ROSCON 2022 Talk are where this skill is structured
Also the gz-sim library under a Apache 2.0 License for modification has allowed for this skill to exist

## More Resources
- Gazebo provides plenty of tutorials on their [HTML Website](https://gazebosim.org/api/sim/10/tutorials.html) 
