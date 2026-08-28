# README images

Screenshots and GIFs used by the top-level `README.md`.

## Needed: `cartpole_gui.png`

A screenshot of the reference CartPole running in the Gazebo GUI, referenced
from the "See it run" section of the main README. Capture it on a machine with
a display and a working graphics driver:

```bash
# Terminal 1 — train the policy (headless, ~1–2 min)
pixi run train

# Terminal 2 — open the Gazebo GUI
pixi run sim

# Terminal 3 — drive the GUI with the trained policy
pixi run deploy --backend harness
```

Once the four poles are balancing, capture the Gazebo window and save it here
as `cartpole_gui.png`. A short screen-recording exported as `cartpole_gui.gif`
is even better for the README: motion sells "it's really simulating" far
better than a still.

Keep images reasonably small (< ~1 MB for the PNG); crop to the simulation
viewport rather than the whole desktop.
