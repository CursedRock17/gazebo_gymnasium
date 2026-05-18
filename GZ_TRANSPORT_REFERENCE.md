# Gazebo Transport (gz-transport13) — Reference for Python RL Integration

Distilled from the gz-transport13 source and tutorials. Covers everything needed
when writing Python environments that communicate with Gazebo Sim.

---

## 1. Core Concepts

### Architecture
Gazebo Transport is **fully distributed** — no central broker. Every node
can publish, subscribe, offer services, and request services simultaneously.
Discovery is automatic via UDP multicast.

### Two Communication Paradigms
| Paradigm | Use when |
|---|---|
| **Pub/Sub** (topics) | Streaming data — joint states, poses, sensor readings |
| **Service calls** | Request/response — world control, set_pose, parameter queries |

---

## 2. Python API Quick Reference

### Installation
```bash
sudo apt install python3-gz-transport13 python3-gz-msgs10
```

### Imports
```python
from gz.transport13 import Node, AdvertiseMessageOptions, SubscribeOptions
from gz.msgs10.double_pb2 import Double
from gz.msgs10.model_pb2 import Model
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.world_control_pb2 import WorldControl
```

### Node Lifecycle
```python
node = Node()   # Create once; reuse across pub/sub/service calls
```
One `Node` per logical role is sufficient. Creating extra Nodes is safe but adds
thread overhead — gz-transport creates multiple background threads per Node.

---

## 3. Publishing

```python
from gz.transport13 import Node, AdvertiseMessageOptions
from gz.msgs10.double_pb2 import Double

node = Node()
pub = node.advertise("/model/cartpole/joint/slider_to_cart/0/cmd_pos", Double,
                     AdvertiseMessageOptions())

msg = Double()
msg.data = 0.5
pub.publish(msg)
```

### Rate Throttling (optional)
```python
opts = AdvertiseMessageOptions()
opts.msgs_per_sec = 100           # cap at 100 Hz
pub = node.advertise(topic, MsgType, opts)
```

---

## 4. Subscribing

Callbacks run on a **background thread** owned by gz-transport. Always protect
shared state with `threading.Lock`.

```python
import threading
from gz.msgs10.model_pb2 import Model

_lock = threading.Lock()
_latest_pos = 0.0

def _on_joint_state(msg: Model):
    global _latest_pos
    with _lock:
        for joint in msg.joint:
            if joint.name == "slider_to_cart":
                _latest_pos = joint.axis1.position

node = Node()
node.subscribe(Model, "/world/cartpole/model/cartpole/joint_state", _on_joint_state)
```

### Rate Throttling (optional)
```python
from gz.transport13 import SubscribeOptions
opts = SubscribeOptions()
opts.msgs_per_sec = 50
node.subscribe(Model, topic, callback, opts)
```

> **Critical:** The GIL does NOT protect you here. gz-transport bindings are
> pybind11 wrappers over C++ threads; they release the GIL. Always use
> `threading.Lock` for any state shared between a callback and the main thread.

---

## 5. Service Calls (Synchronous)

The Python API only supports **requesting** services, not advertising them
(C++ only for the server side).

```python
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.boolean_pb2 import Boolean

node = Node()
req = WorldControl()
req.pause = True

result, response = node.request(
    "/world/cartpole/control",
    req,
    WorldControl,   # request type
    Boolean,        # response type
    timeout=5000    # ms
)
# result: bool — True if the service responded before timeout
# response: Boolean proto — service-defined response payload
```

### Common Gazebo Sim Services
| Service | Request type | Purpose |
|---|---|---|
| `/world/{name}/control` | `WorldControl` | pause / unpause / reset |
| `/world/{name}/set_pose` | `Pose` | teleport any entity |
| `/world/{name}/create` | `EntityFactory` | spawn a model at runtime |
| `/world/{name}/remove` | `Entity` | delete an entity |

### WorldControl Fields
```python
req = WorldControl()
req.pause = True          # pause simulation
req.pause = False         # unpause
req.reset.all = True      # reset all entities to initial state
req.step = 1              # advance exactly N physics steps (blocking)
```

> **Performance note:** `req.step = N` is a blocking service call that holds
> Python for the full N-step duration (~80-100 ms for N=10 at 100 Hz). Use
> unpause → callback counting → pause instead for hot-path stepping.

---

## 6. Key Gazebo Sim Topics

### Joint State (Model proto)
Published by `JointStatePublisher` system — fires every physics step.
```
/world/{world_name}/model/{model_name}/joint_state   →  gz.msgs.Model
```
Access per joint: `msg.joint[i].name`, `.axis1.position`, `.axis1.velocity`

### Dynamic Pose (Pose_V proto)
Published by `SceneBroadcaster` — root body position and orientation.
```
/world/{world_name}/dynamic_pose/info   →  gz.msgs.Pose_V
```
Access: `for pose in msg.pose: if pose.name == "model_name": pose.position.{x,y,z}`

### Force / Torque Command (Double proto)
```
/model/{model_name}/joint/{joint_name}/cmd_force   →  gz.msgs.Double
```
Requires `JointController` with `use_force_commands=true` in the SDF.

### Position Command (Double proto)
```
/model/{model_name}/joint/{joint_name}/0/cmd_pos   →  gz.msgs.Double
```
Requires `JointController` with `use_velocity_commands=false` (default).
Note: the `0` in the topic path is illegal in ROS 2 topic names — cannot be
bridged via `ros_gz_bridge`.

### Clock
```
/clock   →  gz.msgs.Clock
```

---

## 7. GZ_PARTITION — The Most Common Failure Mode

**Default partition:** `<HOSTNAME>:<USERNAME>` (auto-generated per process).
If Gazebo and your Python script use different default partitions, they are
**completely invisible to each other** — subscriptions silently receive nothing,
service calls time out.

**Always set explicitly:**
```bash
export GZ_PARTITION=0   # in every terminal that touches gz-transport
```
Or set it per-process in Python:
```python
import os
os.environ["GZ_PARTITION"] = "0"   # must be set BEFORE creating any Node
```
Or in a ROS 2 launch file:
```python
SetEnvironmentVariable('GZ_PARTITION', '0')
```

---

## 8. Environment Variables Summary

| Variable | Default | Purpose |
|---|---|---|
| `GZ_PARTITION` | `hostname:username` | Isolates topic/service namespaces between process groups. **Set this.** |
| `GZ_IP` | auto | Force a specific NIC when multiple IPs exist |
| `GZ_DISCOVERY_MSG_PORT` | 10317 | UDP port for message discovery |
| `GZ_DISCOVERY_SRV_PORT` | 10318 | UDP port for service discovery |
| `GZ_DISCOVERY_MULTICAST_IP` | 239.255.0.7 | Multicast address for discovery |
| `GZ_TRANSPORT_RCVHWM` | 1000 | Incoming message buffer depth (0 = unlimited) |
| `GZ_TRANSPORT_SNDHWM` | 1000 | Outgoing message buffer depth (0 = unlimited) |
| `GZ_VERBOSE` | 0 | Enable debug logging |
| `GZ_RELAY` | — | Colon-separated IPs to relay discovery across routers |

---

## 9. Topic Naming Rules

Valid: `/topicA`, `/a/b/c`, `topicA` (no leading slash adds one).  
Invalid: empty string, white space, `//double-slash`, `/` alone, `~`, `@`, `:=`.

A **namespace** prefix is applied only to relative topic names (no leading `/`).
Absolute topics (`/foo`) bypass the namespace.

---

## 10. Topic Scope (Visibility)

```python
from gz.transport13 import AdvertiseMessageOptions

opts = AdvertiseMessageOptions()
# opts.scope = ...  # Process, Host, or All (default: All)
```

| Scope | Visible to |
|---|---|
| `Process` | Only nodes in the same OS process |
| `Host` | Nodes on the same machine |
| `All` | Any node on any reachable network (default) |

---

## 11. Hot-Path Stepping Pattern (Preferred)

Avoids the blocking `step=N` service call. Replaces it with:
1. `unpause()` — returns immediately
2. Wait for N joint-state callbacks via `threading.Event`
3. `pause()` — returns immediately

```python
import threading

_phys_lock = threading.Lock()
_phys_steps = 0
_phys_target = 0
_phys_event = threading.Event()
_phys_counting = False

def _on_joint_state(msg):
    global _phys_steps, _phys_counting
    with _phys_lock:
        if not _phys_counting:
            return
        _phys_steps += 1
        if _phys_steps >= _phys_target:
            _phys_event.set()

def _advance_physics(n, world_node, service):
    global _phys_target, _phys_counting
    req = WorldControl(); req.pause = False
    with _phys_lock:
        _phys_target = _phys_steps + n
        _phys_event.clear()
        _phys_counting = True
    world_node.request(service, req, WorldControl, Boolean, 5000)  # unpause
    ok = _phys_event.wait(timeout=5.0)
    req.pause = True
    world_node.request(service, req, WorldControl, Boolean, 5000)  # pause
    with _phys_lock:
        _phys_counting = False
    return ok
```

This is what `GazeboEnv._advance_physics()` implements. The joint-state callback
fires at 100 Hz (physics rate), so waiting for N callbacks = N × 10 ms of
simulated time with no Python-side blocking beyond the event wait.

---

## 12. Discovery Timing

After creating a `Node`, allow ~1 second before making service calls or expecting
subscriptions to be populated. gz-transport uses UDP multicast for discovery;
on loopback (`lo`) this is nearly instant, but DDS/ROS 2 middleware discovery
can take longer.

```python
node = Node()
time.sleep(1.0)   # wait for service advertisement to propagate
```

---

## 13. Debugging

```bash
# List all active topics
gz topic --list

# Echo a topic
gz topic --echo -t /world/cartpole/model/cartpole/joint_state

# List active services
gz service --list

# Verbose transport logging
export GZ_VERBOSE=1
```
