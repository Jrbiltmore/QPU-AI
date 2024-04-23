# /plugins/plugin_interface.py

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class PluginInterface(ABC):
    """
    Abstract base class for all plugins, providing a structured and comprehensive 
    approach to integrating and managing plugins within a larger application framework.
    This interface enforces essential lifecycle methods and setup requirements.
    """
    
    def __init__(self, app: Any, settings: Dict[str, Any]) -> None:
        """
        Initialize the plugin with the application context and specific settings,
        including error handling and logging setup.

        Parameters:
            app (Any): The main application object where the plugin will be registered.
            settings (Dict[str, Any]): Configuration settings specific to the plugin,
                                       allowing for customized plugin behavior.
        """
        self.app = app
        self.settings = settings
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing {self.__class__.__name__} with settings: {settings}")
        super().__init__()

    @abstractmethod
    def load(self) -> None:
        """
        Load the plugin, implementing the necessary setup such as registering routes,
        event handlers, and other integrations with the main application. This method
        should include error handling to manage and log issues during plugin activation.
        """
        try:
            # Example of setup logic
            self.setup_routes()
            self.setup_event_handlers()
            self.logger.info(f"{self.__class__.__name__} loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load {self.__class__.__name__}: {e}")
            raise

    @abstractmethod
    def unload(self) -> None:
        """
        Unload the plugin, carefully cleaning up and unregistering resources used by the plugin.
        This method ensures that the unloading process is thorough, avoiding resource leakage and
        maintaining system integrity.
        """
        try:
            # Example of cleanup logic
            self.cleanup_routes()
            self.cleanup_event_handlers()
            self.logger.info(f"{self.__class__.__name__} unloaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to unload {self.__class__.__name__}: {e}")
            raise

    def setup_routes(self) -> None:
        """
        Setup web routes specific to this plugin. This method demonstrates how to dynamically add
        routes to the Flask application that is managing the plugin.
    
        Each route is configured to handle specific endpoints that are relevant to the plugin's functionality,
        demonstrating a clear separation of concerns and modular design.
        """
        if hasattr(self.app, 'add_url_rule'):
            # Example of adding a new route for a plugin-specific feature
            self.app.add_url_rule(
                '/plugin_specific_endpoint',         # Endpoint URL
                'plugin_specific_endpoint',          # Endpoint name
                self.plugin_specific_handler,        # View function handling this endpoint
                methods=['GET', 'POST'],             # Allowed methods
                strict_slashes=False
            )
            self.logger.info(f"Routes for {self.__class__.__name__} set up successfully.")
        else:
            self.logger.error(f"Failed to setup routes: App object does not support route additions.")
            raise NotImplementedError("App object must support dynamic route additions.")
    
    def plugin_specific_handler(self):
        """
        Handle requests for the plugin-specific endpoint. This method acts as a view function
        for the endpoint defined in setup_routes.
    
        Returns:
            str: A response string or data to be sent back to the client.
        """
        return "Response from plugin-specific endpoint"
    
    def setup_event_handlers(self) -> None:
        """
        Setup event handlers specific to this plugin. This is a placeholder for actual event handling setup.
        """
        pass

    def cleanup_routes(self) -> None:
        """
        Clean up all routes established by this plugin. This is a placeholder for actual route cleanup.
        """
        pass

    def cleanup_event_handlers(self) -> None:
        """
        Clean up all event handlers established by this plugin. This is a placeholder for actual event handler cleanup.
        """
        pass
