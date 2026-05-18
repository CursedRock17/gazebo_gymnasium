## Tasks to Execute
I'll be away from the keyboard, thus you have permissions to execute any program given it's not destructive. If you cannot make a decisino, simple make a note in this file and move on. Don't execute anything that coudl be harmful, but don't ever stop to ask for validation, as I cannot give it.

You have permissions to serach through files on the device, access the internet, run shell comamands

## Current Tasks
- [ ] After about 30 seconds of wiaitng to load, I get a Warning: "_advance_phsyics timed out after 5.0s"
- [ ] Multiple other libraries benefit speed wise, from being able to run directly in the simulation with the bridge integrated in, is that something feasbile here or will we be constrained to a DDS-enabled environment.
- [ ] Make the ROSBag2 logging and Foxglove Dispaly optional with their defaults as false
- [ ] Reduce FPS losses from Latency
- [ ] Review all Gazebo Sim Tutorials, Look for ways to save compute: https://gazebosim.org/api/sim/10/tutorials.html
- [ ] Review all Gazebo Transport Tutorials, Look for ways to save compute: https://gazebosim.org/api/transport/13/tutorials.html
- [ ] Take a look at the `gz-transport` directory in this project, draft important pieces of information that are needed when using the gazebo transport system. Nicely header and label them, add theme as a markdown here. When finished !`rm -rf gz-transport`
- [ ] Since this is going to be a repository that is going to be distributed, more of the work should be on us instead of the end user. If there's repeated logic in multiple examples, that is agnostic of any system/RL library, it should be moved into the base repository
- [ ] You should be able to create a custom versino of PPO and run that instead of SB3, but in the same format. This will be a base ground for testing if non-SB3 linbarries or custom code will work with this framework.
- [ ] Update Unit Tests to be inlinew ith all of the recent code chagnes, then run the tests again to make sure they fpass, if they fail, makes necessary changes and replay




## Completion Notes
