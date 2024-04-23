
# /Data_Handling_Processing/persistent_memory_manager.py

import os
import mmap

class PersistentMemoryManager:
    def __init__(self, filename, size):
        self.filename = filename
        self.size = size
        self.fd = os.open(filename, os.O_RDWR | os.O_CREAT, 0o666)
        os.ftruncate(self.fd, size)
        self.memory_map = mmap.mmap(self.fd, size, access=mmap.ACCESS_WRITE)

    def write_data(self, data, offset=0):
        # Write data to memory-mapped file at a given offset
        self.memory_map.seek(offset)
        self.memory_map.write(data.encode('utf-8'))

    def read_data(self, size, offset=0):
        # Read data from memory-mapped file at a given offset
        self.memory_map.seek(offset)
        return self.memory_map.read(size).decode('utf-8')

    def close(self):
        # Clean up
        self.memory_map.close()
        os.close(self.fd)
        os.unlink(self.filename)  # Optionally remove the file on close

# Example usage
if __name__ == '__main__':
    pm_manager = PersistentMemoryManager('example.dat', 1024)  # 1KB file
    pm_manager.write_data('Hello, persistent world!', 0)
    print(pm_manager.read_data(25, 0))
    pm_manager.close()
