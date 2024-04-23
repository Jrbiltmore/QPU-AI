# plugin_manager.py

class PluginManager:
    """
    Manages plugins for the system.
    """

    def __init__(self):
        """
        Initialize the PluginManager instance.
        """
        self.plugins = []

    def add_plugin(self, plugin):
        """
        Add a plugin to the manager.

        Parameters:
            plugin: The plugin instance to add.
        """
        self.plugins.append(plugin)

    def remove_plugin(self, plugin):
        """
        Remove a plugin from the manager.

        Parameters:
            plugin: The plugin instance to remove.
        """
        self.plugins.remove(plugin)

    def get_plugins(self):
        """
        Get the list of plugins managed by the manager.

        Returns:
            list: The list of plugins.
        """
        return self.plugins
