# Rover-side runtime

`rover_line_deploy.py` belongs in the rover repository, not here: no physical
logic lives in `gazebo_gymnasium`. It is kept in this directory as the
reference copy of the deployment side, so the two halves of the contract can be
reviewed together.

It imports `rover_line_contract`, which is **not** in this directory on
purpose. The contract has exactly two homes — the canonical one at
`gazebo_gymnasium_bridge/gazebo_gymnasium_bridge/envs/rover_line_contract.py`,
and a copy in the rover repo beside the runtime. Keeping a third copy here
would mean three things to keep in step, and a divergent extractor does not
fail at the boundary: it fails on hardware, in a way that still looks
plausible.

To deploy:

```bash
cp <gazebo_gymnasium>/gazebo_gymnasium_bridge/gazebo_gymnasium_bridge/envs/rover_line_contract.py \
   <rover-repo>/rover_line_contract.py
cp <gazebo_gymnasium>/results/for_rover_repo/rover_line_deploy.py <rover-repo>/
```

Then, on a machine that can import the simulator package, check the copy
against the original before trusting it:

```bash
python rover_line_deploy.py --self-check
```

That compares behaviour rather than bytes — the feature extractor, the
termination test and the action map are diffed against the simulator's own on
200 randomised frames, and the contract's `SCHEMA_VERSION` must match. A
policy trained against a different layout otherwise loads without complaint
and then acts on mis-sliced floats.
