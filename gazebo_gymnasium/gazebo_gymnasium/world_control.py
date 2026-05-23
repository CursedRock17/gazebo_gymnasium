import time

_TIMEOUT_MS = 5000
_RETRY_DELAY_S = 0.1
_MAX_RETRIES = 3


class WorldController:
    """Drives a Gazebo world via the WorldControl service over gz.transport.

    In continuous mode (the default), Gazebo runs at unlimited speed after the
    initial ping/start.  Individual physics steps are synchronized in GazeboEnv
    via joint-state callback counting — no blocking service calls in the inner
    loop.  Service calls are only made at episode boundaries (reset) and at
    startup (ping/start).
    """

    def __init__(self, world_name: str, steps_per_action: int = 10):
        from gz.transport13 import Node
        from gz.msgs10.world_control_pb2 import WorldControl
        from gz.msgs10.boolean_pb2 import Boolean

        self._WorldControl = WorldControl
        self._Boolean = Boolean
        self._node = Node()
        self.steps_per_action = steps_per_action
        self._service = f"/world/{world_name}/control"

    def _request(self, req) -> bool:
        for attempt in range(_MAX_RETRIES):
            result, _ = self._node.request(
                self._service, req, self._WorldControl, self._Boolean, _TIMEOUT_MS
            )
            if result:
                return True
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY_S)
        return False

    def step(self, n: int) -> bool:
        """Advance the simulation by exactly n physics steps, then auto-pause.

        Sends WorldControl(pause=True, multi_step=n).  The pause=True flag is
        REQUIRED — Gazebo's ProcessWorldControl calls SetPaused(control.pause)
        FIRST, then checks `if (Paused() && multiStep > 0)`.  If pause is omitted
        it defaults to False in proto3, which unpauses the simulation and makes
        the multi_step condition fail, leaving the sim running freely.

        With pause=True, Gazebo:
          1. Sets paused=True (condition met)
          2. Adds n to pendingSimIterations
          3. Internally unpauses to run exactly n steps
          4. Auto-pauses after the nth step via pendingSimIterations countdown

        Source: gz-sim SimulationRunner.cc ProcessWorldControl()
        """
        req = self._WorldControl()
        req.pause = True   # required: makes Paused() True before multi_step check
        req.multi_step = n  # Gazebo then internally unpauses and runs n steps
        result, _ = self._node.request(
            self._service, req, self._WorldControl, self._Boolean, _TIMEOUT_MS
        )
        return result

    def ping(self) -> bool:
        """Verify connectivity by pausing Gazebo.  Leaves the sim PAUSED so that
        _advance_physics() in GazeboEnv controls all stepping via step()."""
        req = self._WorldControl()
        req.pause = True
        return self._request(req)

    def reset(self) -> bool:
        """Reset all entities to initial state and leave Gazebo PAUSED.

        Sending both reset.all and pause=True in a single service call ensures
        Gazebo is deterministically paused at the initial state when this call
        returns.  Without pause=True, Gazebo may start running immediately after
        reset, causing joint-state callbacks to fire before _advance_physics()
        starts counting — leading to timeouts and stale observations.
        """
        req = self._WorldControl()
        req.reset.all = True
        req.pause = True  # critical: leave sim paused so step counting starts clean
        result = self._request(req)
        if not result:
            print(f"[WARN] WorldController.reset() failed after {_MAX_RETRIES} attempts"
                  f" — service={self._service}", flush=True)
        return result

    def pause(self) -> bool:
        req = self._WorldControl()
        req.pause = True
        result, _ = self._node.request(
            self._service, req, self._WorldControl, self._Boolean, _TIMEOUT_MS
        )
        return result

    def unpause(self) -> bool:
        req = self._WorldControl()
        req.pause = False
        result, _ = self._node.request(
            self._service, req, self._WorldControl, self._Boolean, _TIMEOUT_MS
        )
        return result
