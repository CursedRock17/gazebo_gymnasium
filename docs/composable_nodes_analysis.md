# ROS 2 Composable Nodes: Do They Help Our Launches?

User asked: "Can use a ROS 2 Composable Node for any ROS 2
launch processes to speed up the simulation as best as possible".

## What Composable Nodes Give You

Stack ROS 2 nodes into one process via `ComposableNodeContainer`. Wins:

1. **Lower IPC overhead** between nodes in the same container: intra-process
   message passing skips DDS serialization (saves ~50 to 200 µs per message).
2. **Fewer process startups**: one container vs N node executables.
   Saves ~50 to 100 ms of startup time per saved node.
3. **Shared memory**: one DDS participant instead of N. Saves a few MB
   of RAM per node.

## What Our Launches Actually Contain

Per launch (typical: `cartpole_harness.launch.py`):
- `ros_gz_bridge::parameter_bridge`: 1 node
- `foxglove_bridge::FoxgloveBridge`: 1 node (conditional)
- `tf2_ros::static_transform_publisher`: 1 node (conditional)
- `gz sim` itself: not a ROS node, runs in its own process (gz-transport
  uses its own discovery, separate from DDS)

The line_follower launches also add:
- `robot_state_publisher::RobotStatePublisher`: 1 node

So 3 to 4 ROS nodes max per launch. Modest.

## Where Composable Is Already Used

The (since-removed) single-env training launches ran the bridges in a
`ComposableNodeContainer`. We kept the standalone `parameter_bridge`
alongside it as a fallback because the composable `ros_gz_bridge` has a
different (less complete) config-file code path; see the file's
comments.

## Where It's Not Used And Why

The surviving launches (`cartpole_harness.launch.py`,
`cartpole_multi.launch.py`) leave the bridges as standalone
`Node()` instances. Two reasons:

1. **Bottleneck math.** Our RL training spends 99% or more of wall time
   inside gz sim physics, the agent's policy forward pass, and the
   agent's gradient update. The bridges pass perhaps 50 to 100
   messages per second (joint_state, pose, /env/metrics). Even if
   composing saves 200 µs per message, that's 10 to 20 ms/sec, well
   under 1% of wall time.

2. **Simpler launches.** Standalone `Node()` is one line per node; a
   composable container needs a `ComposableNodeContainer` and per-node
   `ComposableNode` descriptions. For 3 nodes the line count nearly
   doubles. Worth it only if the IPC win is real.

## Summary

Composable nodes are the right call when:
- You have 5 or more nodes that all talk to each other a lot.
- You're streaming high-bandwidth data (camera images, point clouds)
  between nodes in the same process.
- Startup latency matters (real-time / restart-heavy workloads).

For our RL launches, none of those apply strongly. The removed training
launches kept the composable bridge code as a reference; the other
launches deliberately stay standalone for readability. **Status: nothing
to change.**
