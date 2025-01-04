"""
This script demonstrates both synchronous and asynchronous approaches to handling
data streams and persistence in Python using file handling.
"""

import asyncio
import aiofiles  # For async file operations
from typing import AsyncIterator, Iterator
from pathlib import Path

class PersistentStorage:
    """Custom storage class for both sync and async operations."""
    
    def __init__(self, file_path: str):
        """Initialize storage with file path."""
        self.file_path = Path(file_path)
        
    def save(self, data: str) -> None:
        """Synchronous save operation."""
        with open(self.file_path, 'a') as file:
            file.write(f"{data}\n")
            
    async def async_save(self, data: str) -> None:
        """Asynchronous save operation."""
        async with aiofiles.open(self.file_path, 'a') as file:
            await file.write(f"{data}\n")

class DataStreamHandler:
    """Handles data stream generation in both sync and async patterns."""
    
    def generate_data_stream(self) -> Iterator[str]:
        """Synchronous data stream generator."""
        for i in range(10):
            yield f"Message {i}"

    async def generate_async_data_stream(self) -> AsyncIterator[str]:
        """Asynchronous data stream generator."""
        for i in range(10):
            await asyncio.sleep(0.1)  # Simulate async data generation
            yield f"Message {i}"

class StorageManager:
    """Manages the persistence of data using both sync and async methods."""
    
    def __init__(self, sync_path: str = "sync_data.txt", async_path: str = "async_data.txt"):
        """Initialize storage handlers."""
        self.sync_storage = PersistentStorage(sync_path)
        self.async_storage = PersistentStorage(async_path)

    def synchronous_stream_and_persist(self) -> None:
        """Handle synchronous streaming and persistence."""
        try:
            print("\nSynchronous Streaming:")
            stream = DataStreamHandler()
            
            for message in stream.generate_data_stream():
                print(f"Processing: {message}")
                self.sync_storage.save(message)
            
            print("Data saved synchronously!")
            
        except Exception as e:
            print(f"Error in synchronous processing: {str(e)}")
            raise

    async def asynchronous_stream_and_persist(self) -> None:
        """Handle asynchronous streaming and persistence."""
        try:
            print("\nAsynchronous Streaming:")
            stream = DataStreamHandler()
            
            async for message in stream.generate_async_data_stream():
                print(f"Processing: {message}")
                await asyncio.sleep(0.5)  # Simulate async I/O
                await self.async_storage.async_save(message)
            
            print("Data saved asynchronously!")
            
        except Exception as e:
            print(f"Error in asynchronous processing: {str(e)}")
            raise

async def main():
    """Main function to run both examples."""
    storage_manager = StorageManager()
    
    try:
        # Run synchronous operation
        storage_manager.synchronous_stream_and_persist()
        
        # Run asynchronous operation
        await storage_manager.asynchronous_stream_and_persist()
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)