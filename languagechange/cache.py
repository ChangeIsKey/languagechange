"""Cache manager with atomic write helpers for file-based caches."""

import os
from contextlib import contextmanager
import filelock

DEFAULT_CACHE_DIR = "./cache"  # Default directory for cache files

class CacheManager:
    """
    Manages cache files with atomic write operations to prevent data corruption 
    in concurrent environments. The cache files are saved to a directory that 
    can be specified during initialization.
    """

    def __init__(self, cache_dir=None):
        """
        Initializes the CacheManager with a directory to store cached files.

        Args:
            cache_dir (str, optional): The directory where cache files will be stored. 
                                       Defaults to './cache' if not provided.
        """
        self.cache_dir = os.path.abspath(cache_dir or DEFAULT_CACHE_DIR)
        os.makedirs(self.cache_dir, exist_ok=True)

    @contextmanager
    def atomic_write(self, path):
        """
        Provides a context manager for writing to cache files in an atomic way.
        This ensures that partial writes do not corrupt the target file, 
        especially when multiple processes or threads access the same file.

        Args:
            path (str): The relative path to the cache file within the cache directory.

        Yields:
            file object: A writable file object for writing data. The file is temporary 
                         and will be renamed to the target file path after the write 
                         operation completes successfully.
        """
        full_path = os.path.join(self.cache_dir, path)
        temp_path = full_path + ".tmp"  # Temporary file path
        lock_path = full_path + ".lock"  # Lock file path to prevent concurrent writes

        # Ensure the directory structure exists for the cache file
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # Use a file lock to synchronize access across multiple processes or threads
        with filelock.FileLock(lock_path):
            try:
                # Step 1: Write data to a temporary file
                with open(temp_path, 'wb') as f:
                    # Provide the temporary file handle to the caller for writing
                    yield f  

                # Step 2: Atomically replace the existing file with the new file
                if os.path.exists(full_path):
                    os.remove(full_path)  # Remove the existing file if it exists
                os.rename(temp_path, full_path)  # Rename temp file to target file path

            finally:
                # Cleanup: Remove any leftover temporary or lock files
                for p in [temp_path, lock_path]:
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass  # Ignore errors during cleanup to avoid interference
