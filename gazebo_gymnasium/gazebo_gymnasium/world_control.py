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

    def ping(self) -> bool:
        """Verify connectivity by pausing Gazebo.  Leaves the sim PAUSED so that
        _advance_physics() in GazeboEnv controls all stepping via unpause/pause."""
        req = self._WorldControl()
        req.pause = True
        return self._request(req)

    def reset(self) -> bool:
        """Reset all entities to initial state (blocking — episode boundary only)."""
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
