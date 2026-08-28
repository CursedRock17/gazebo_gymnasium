# Importing CAD Models: URDF To SDF

[Writing The Model SDF](creating_your_own_agent.md#writing-the-model-sdf)
covers authoring SDF by hand, the path every built-in classic-control and
MuJoCo-ported specification uses. Starting from an existing CAD model
instead is often faster once real geometry, not primitive shapes,
matters, the same reason
[ROS 2's own documentation](https://docs.ros.org/en/jazzy/Capabilities/Simulation/URDF/Exporting-an-URDF-File.html)
frames CAD export as the preferred alternative to hand-writing a
Unified Robot Description Format (URDF) file: "Instead of crafting a
URDF by hand it is possible to export a URDF model from many different
CAD and modeling programs." This project has one real precedent,
`models/rover`, exported from Onshape and carried through the exact
pipeline below; everything here is the actual, tested process, not
theory.

## Step 1: Export A URDF From Your CAD Model

ROS 2's own list of CAD exporters, reproduced here since it is the
authoritative, actively maintained source, covers most common tools:

| CAD Program | Exporter |
|---|---|
| Onshape | [OnShape URDF Exporter](https://github.com/Rhoban/onshape-to-robot) (`onshape-to-robot`) |
| SolidWorks | [SolidWorks URDF Exporter](https://github.com/ros/solidworks_urdf_exporter) |
| Fusion 360 | [Fusion 360 URDF Exporter](https://github.com/dheena2k2/fusion2urdf-ros2), or [fusion2URDF](https://github.com/Adriaeik/fusion2URDF) for `ros2_control`/closed loops |
| FreeCAD | [FreeCAD ROS Workbench](https://github.com/galou/freecad.cross), or [RobotCAD](https://github.com/drfenixion/freecad.overcross) |
| CREO Parametric | [creo2urdf](https://github.com/icub-tech-iit/creo2urdf) |
| Multiple (Fusion360, OnShape, Solidworks) | [ExportURDF Library](https://github.com/daviddorf2023/ExportURDF) |

The full, current list, including Blender and other conversion and
viewing tools, lives at the
[ROS 2 page linked above](https://docs.ros.org/en/jazzy/Capabilities/Simulation/URDF/Exporting-an-URDF-File.html);
none of these exporters are ROS-core-maintained, so treat each as
third-party. `models/rover/rover.urdf`'s own header comment carries the
source CAD URL for reference. This project used `onshape-to-robot`, and
every gotcha below was found through that specific export, though most
generalize to any exporter, since they come from the URDF-to-SDF
conversion step, not from Onshape specifically.

## Step 2: Convert URDF To SDF

```bash
gz sdf -p your_model.urdf > model.sdf
```

`gz sdf -p` reads from a real file path, not stdin, so piping a URDF
into it does not work. This one command is the entire conversion; the
gotchas below are all about what the exporter and the converter get
wrong on the way there and what still needs fixing afterward.

## Step 3: Real Gotchas This Repo Has Hit

Every item here comes from actually running the pipeline on the rover
model, not from documentation elsewhere.

- **Mesh URI paths rarely resolve as exported.** `onshape-to-robot`
  wrote `package://assets/meshes/left_wheel.stl`, but the real path in
  this project's package layout is
  `package://gazebo_gymnasium_resources/models/rover/assets/left_wheel.stl`,
  no `meshes/` subdirectory at all. Check every `<mesh filename="...">`
  in the exported URDF against where the mesh files actually landed
  before converting, since a broken URI fails silently at spawn time
  rather than at conversion time.
- **`gz sdf -p` emits phantom `<frame>` blocks.** One per joint, plus
  floor and world anchors, from the URDF-to-SDF graph-building pass.
  Their names often collide with real joint names (`bthigh` in
  HalfCheetah, `pole` in InvertedDoublePendulum were real collisions
  hit while porting the MuJoCo environments through the same
  converter), which fails `gz sdf --check` with a FrameAttachedToGraph
  cycle error. They serve no purpose at runtime.
  [`scripts/fix_sdf_phantom_frames.py`](../scripts/fix_sdf_phantom_frames.py)
  strips them from every model SDF in the repo in one pass and is
  idempotent, safe to rerun.
- **The SDF version tag needs pinning.** `gz sdf -p` stamps whatever
  version the installed `gz-sdformat` defaults to; this project targets
  `1.11` throughout, so the tag needs rewriting to match after
  conversion.
- **`gz sdf -p` produces `model.sdf` only, never `model.config`.**
  Gazebo needs both to load a model by name. Write `model.config` by
  hand; the block below, adapted from `models/rover/model.config`, is
  the minimum:

  ```xml
  <?xml version="1.0"?>
  <model>
    <name>your_model</name>
    <version>1.0</version>
    <sdf version="1.11">model.sdf</sdf>
    <author>
      <name>Your Name</name>
      <email>you@example.com</email>
    </author>
    <description>One line on where this model came from and how it was converted.</description>
  </model>
  ```

- **Mirrored assemblies can export opposite-handed joint axes.** The
  rover's `left_axle` and `right_axle` joints came out of the Onshape
  export with opposite-handed rotation axes, an artifact of the two
  wheels being physically mirrored halves of one assembly. A single
  `DiffDrive` command sent to both then spun one wheel forward and the
  other backward, so the rover span in place instead of driving.
  Nothing about SDF or `gz sdf -p` catches this; it only shows up as
  the model visibly doing the wrong thing in the GUI. Any CAD assembly
  built from a mirrored or patterned symmetric pair is worth checking
  for this specifically.
- **Collision geometry inherited from CAD is usually too expensive to
  keep everywhere.** A CAD export's collision mesh is the actual part
  geometry, exact but costly to resolve contact against at every
  physics step. The rover's conversion dropped mesh collisions
  entirely on parts that live inside a sealed enclosure and can never
  touch anything (motors, brackets, the battery, the electronics), and
  replaced the three wheels' mesh collisions with sphere primitives,
  since wheel-ground contact is the only contact that actually matters
  for that model and a sphere resolves far more cheaply than a mesh.
  Decide per part whether its exact shape needs to participate in
  physics at all.
- **Every `<collision>` still needs a `<visual>` sibling**, the same
  rule [Writing The Model SDF](creating_your_own_agent.md#writing-the-model-sdf)
  states for hand-written SDF: a link with collision but no visual
  loads physically but renders invisibly. CAD exporters that carry
  real geometry usually populate both already, but this is worth
  checking, since one MuJoCo model's converter
  (`mjcf2urdf`, a different pipeline than CAD export, but it fails the
  same way for the same reason) emitted collision only, requiring
  [`scripts/add_visuals_to_sdfs.py`](../scripts/add_visuals_to_sdfs.py)
  as a fix.

## Step 4: Trim To A Bare Model, Or Add Actuation

Once `model.sdf` loads cleanly, the model is a normal Gazebo model, not
yet an `AgentSpec`-ready one. Everything in
[Writing The Model SDF](creating_your_own_agent.md#writing-the-model-sdf)
applies from here identically to a hand-written model: for the harness
backend, strip `<effort>` limits from every joint meant for actuation
(an effort limit silently disables ECM velocity control in this DART
build) and remove any `JointController`/`JointStatePublisher`, since the
world-level harness plugin owns actuation and sensing instead; every
actuated or observed joint needs a name the `AgentSpec` can reference by
name; the model should not pin itself to the world inside `model.sdf`,
since the spawner adds that through `extra_joints`. The rover model
instead kept a `DiffDrive` plugin and stayed a controller-equipped,
per-agent-backend model, the other valid path, matching
[the three backends](creating_your_own_agent.md#the-three-backends)'
own tradeoffs.

## Worked Example: The Rover

[`scripts/convert_rover_urdf.py`](../scripts/convert_rover_urdf.py) is
the complete, real pipeline for this project's one CAD-sourced model,
worth reading end to end as a template. In order: fix the mesh URIs,
run `gz sdf -p` against a tempfile copy of the fixed URDF, pin the SDF
version, strip phantom frames, drop the internal-part collisions by
matching each collision block's embedded mesh URI against a
disposition list (robust to the URDF's part ordering shifting on a
re-export), inject sphere collisions for the three wheels, flip
`right_axle`'s rotation axis, rename the model, then append the
`DiffDrive`/`JointStatePublisher` plugin block and a forward-facing
camera link before the closing `</model>` tag. Every step is a plain
text transform on the SDF, regex-based rather than a DOM rewrite,
deliberately, since round-tripping through `ElementTree` risks losing
formatting the project doesn't otherwise touch. Adapting this script for
a new CAD-sourced model means changing the disposition list, the wheel
or actuator geometry, and whatever plugins the new model needs, not
rewriting the pipeline itself.

## Checklist

- [ ] URDF exported from CAD, mesh files present at the paths the URDF
      actually references.
- [ ] `gz sdf -p your_model.urdf > model.sdf` runs without error.
- [ ] SDF version pinned to `1.11`.
- [ ] Phantom `<frame>` blocks stripped
      (`scripts/fix_sdf_phantom_frames.py`).
- [ ] `model.config` written by hand.
- [ ] `gz sdf --check model.sdf` passes, no FrameAttachedToGraph errors.
- [ ] Every `<collision>` has a matching `<visual>`.
- [ ] Symmetric/mirrored parts checked for opposite-handed joint axes.
- [ ] Expensive mesh collisions on non-contact parts reviewed, dropped
      or simplified where physics never needs them.
- [ ] Loads and renders correctly in the Gazebo GUI before continuing
      to [Defining The AgentSpec](creating_your_own_agent.md#defining-the-agentspec).

## Glossary

| Term | Definition |
|---|---|
| **Computer-Aided Design (CAD)** | The 3D modeling software a model's real geometry originates from |
| **Unified Robot Description Format (URDF)** | The XML format most CAD exporters and ROS tooling produce; this project's actual model format is SDF, so URDF is always an intermediate step |
| **Simulation Description Format (SDF)** | Gazebo's native model and world format; every model in this project ships as SDF, whether hand-written or converted |
| **FrameAttachedToGraph** | The SDF validation pass that fails when a `<frame>` element's name collides with another element in the model's frame graph, the failure mode phantom frames from `gz sdf -p` trigger |
