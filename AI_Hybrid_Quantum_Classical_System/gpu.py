# gpu.py

class GPU:
    """
    Represents a GPU (Graphics Processing Unit) device.
    """

    def __init__(self, name: str, memory_size: int):
        """
        Initialize a GPU instance with the specified name and memory size.

        Parameters:
            name (str): The name or model of the GPU.
            memory_size (int): The memory size of the GPU in megabytes (MB).
        """
        self.name = name
        self.memory_size = memory_size
        self.temperature = 0
        self.usage = 0

    def update_temperature(self, new_temperature: float):
        """
        Update the temperature of the GPU.

        Parameters:
            new_temperature (float): The new temperature value in degrees Celsius.
        """
        self.temperature = new_temperature

    def update_usage(self, new_usage: float):
        """
        Update the usage of the GPU.

        Parameters:
            new_usage (float): The new usage value as a percentage.
        """
        self.usage = new_usage

    def get_name(self) -> str:
        """
        Get the name of the GPU.

        Returns:
            str: The name of the GPU.
        """
        return self.name

    def get_memory_size(self) -> int:
        """
        Get the memory size of the GPU.

        Returns:
            int: The memory size of the GPU in megabytes (MB).
        """
        return self.memory_size

    def get_temperature(self) -> float:
        """
        Get the temperature of the GPU.

        Returns:
            float: The temperature of the GPU in degrees Celsius.
        """
        return self.temperature

    def get_usage(self) -> float:
        """
        Get the usage of the GPU.

        Returns:
            float: The usage of the GPU as a percentage.
        """
        return self.usage
