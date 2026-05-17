import time

_TIMEOUT_MS = 5000
_RETRY_DELAY_S = 0.1
_MAX_RETRIES = 3


class WorldController:
    """Drives a Gazebo world via the WorldControl service over gz.transport."""

    def __init__(self, world_name: str, steps_per_action: int = 10):
        # Imported here so the package can be imported without Gazebo installed
        # (allows unit tests and static analysis to work in any environment).
        from gz.transport13 import Node
        from gz.msgs10.world_control_pb2 import WorldControl
        from gz.msgs10.boolean_pb2 import Boolean

        self._WorldControl = WorldControl
        self._Boolean = Boolean
        self._node = Node()
        self._steps_per_action = steps_per_action
        self._service = f"/world/{world_name}/control"

    def _request(self, req) -> bool:
        """Send a WorldControl request, retrying up to _MAX_RETRIES times."""
        for attempt in range(_MAX_RETRIES):
            result, _ = self._node.request(
                self._service, req, self._WorldControl, self._Boolean, _TIMEOUT_MS
            )
            if result:
                return True
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY_S)
        return False

    def ping(self) -> bool:
        """Check connectivity by attempting a no-op pause→unpause round-trip.

        Call this at startup to verify the WorldControl service is reachable
        before beginning training.  Returns True if the service responds.
        Uses the same retry logic as _request() because a freshly-created
        gz.transport Node may need several attempts before service discovery
        completes.
        """
        req = self._WorldControl()
        req.pause = True
        if not self._request(req):
            return False
        # Unpause best-effort; failure here is non-fatal
        req2 = self._WorldControl()
        req2.pause = False
        self._node.request(
            self._service, req2, self._WorldControl, self._Boolean, _TIMEOUT_MS
        )
        return True

    def step(self) -> bool:
        """Step the simulation by steps_per_action physics ticks, then pause."""
        req = self._WorldControl()
        req.multi_step = self._steps_per_action
        result = self._request(req)
        if not result:
            print(f"[WARN] WorldController.step() failed after {_MAX_RETRIES} attempts"
                  f" — service={self._service}", flush=True)
        return result

    def reset(self) -> bool:
        """Reset all entities in the world to their initial state."""
        req = self._WorldControl()
        req.reset.all = True
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
