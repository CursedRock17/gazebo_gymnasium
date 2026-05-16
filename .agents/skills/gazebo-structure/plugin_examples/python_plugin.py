# Standard Modules
from typing import Optional

# Gazebo Modules
from gz.sim8 import Model, Link, World, world_entity


class GazeboPlugin():
    """
    Main class to integrate plugin into a Gazebo simulation.
    """

    def configure(self, entity, element, ecm, eventManager):
        """
        Necessary function for Gazebo plugin on startup, called once sim time has begun.
        Args:
            entity: Gazebo "entity" which is essentially whatever the plugin needs to access,
            in this case, it's accessing the world.
            element: SDF representation of the world.
            ecm: Entity Component Manager, keeps track of individual tags within
            the SDF file, thus the world.
            eventManager: Monitors all occuring actions in the world.
        """
        # Establish Model Info
        self.model = Model(entity)
        self.link = Link(self.model.canonical_link(ecm))
        print("Configured Plugin For: ", entity)

        # Establish World Info
        world_ent = world_entity(ecm)
        world = World(world_ent)
        self.world_name = world.name(ecm)

    def pre_update(self, info, ecm):
        """
        Necessary function for Gazebo plugin before the starting of each scene.
        Modifies state before physics run, important for motor control.
        Args:
            info: Information about the current scene.
            ecm: Entity Component Manager, keeps track of individual tags within
            the SDF file, thus the world.
        """
        if info.paused or self.needs_reset:
            return

    def update(self, info, ecm):
        """
        Necessary function for Gazebo plugin to run during each scene.
        Used for physics simulation step.
        Args:
            info: Information about the current scene.
            ecm: Entity Component Manager, keeps track of individual tags within
            the SDF file, thus the world.
        """
        if info.paused or self.needs_reset:
            return

    def post_update(self, info, ecm):
        """
        Necessary function for Gazebo plugin to run after each scene.
        Used for post physics processing step: reading results for sensors.
        Args:
            info: Information about the current scene.
            ecm: Entity Component Manager, keeps track of individual tags within
            the SDF file, thus the world.
        """
        if info.paused:
            return

    def reset(self, info, ecm):
        """
        Gazebo system plugin: reset(info, ecm)
        """
        self.needs_reset = False


def get_system():
    """Allows Gazebo Python Path to find this file"""
    return GazeboPlugin()
