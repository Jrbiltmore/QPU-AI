# qpu.py

class QPU:
    """
    Represents a QPU (Quantum Processing Unit) device.
    """

    def __init__(self, name: str, logical_qubits: int):
        """
        Initialize a QPU instance with the specified name and number of logical qubits.

        Parameters:
            name (str): The name or model of the QPU.
            logical_qubits (int): The number of logical qubits available in the QPU.
        """
        self.name = name
        self.logical_qubits = logical_qubits
        self.temperature = 0
        self.usage = 0

    def update_temperature(self, new_temperature: float):
        """
        Update the temperature of the QPU.

        Parameters:
            new_temperature (float): The new temperature value in degrees Celsius.
        """
        self.temperature = new_temperature

    def update_usage(self, new_usage: float):
        """
        Update the usage of the QPU.

        Parameters:
            new_usage (float): The new usage value as a percentage.
        """
        self.usage = new_usage

    def get_name(self) -> str:
        """
        Get the name of the QPU.

        Returns:
            str: The name of the QPU.
        """
        return self.name

    def get_logical_qubits(self) -> int:
        """
        Get the number of logical qubits in the QPU.

        Returns:
            int: The number of logical qubits available in the QPU.
        """
        return self.logical_qubits

    def get_temperature(self) -> float:
        """
        Get the temperature of the QPU.

        Returns:
            float: The temperature of the QPU in degrees Celsius.
        """
        return self.temperature

    def get_usage(self) -> float:
        """
        Get the usage of the QPU.

        Returns:
            float: The usage of the QPU as a percentage.
        """
        return self.usage
