"""NPZ file key naming and management"""
from enum import Enum
from typing import Optional


class KeyType(Enum):
    """Enumeration of NPZ array key types"""
    IMAGE = 'image'
    MASK = 'mask'


class KeyManager:
    """Manages key names for NPZ array storage and retrieval"""
    
    # Single window keys (backward compatible with server_lux)
    SINGLE_WINDOW_IMAGE_KEY = 'image'
    SINGLE_WINDOW_MASK_KEY = 'mask'
    
    # Multi-window key separator
    MULTI_WINDOW_SEPARATOR = '_'
    
    # Dispatcher map for single-window keys
    _SINGLE_WINDOW_KEYS = {
        KeyType.IMAGE: SINGLE_WINDOW_IMAGE_KEY,
        KeyType.MASK: SINGLE_WINDOW_MASK_KEY,
    }
    
    @classmethod
    def get_image_key(cls, window_id: Optional[str] = None) -> str:
        """
        Get NPZ key for image array.
        
        Args:
            window_id: Window identifier. If None, returns single-window key.
                      If provided, returns multi-window key with window_id prefix.
        
        Returns:
            str: Key name for accessing image in NPZ file
        
        Examples:
            >>> KeyManager.get_image_key()
            'image'
            >>> KeyManager.get_image_key('window_1')
            'window_1_image'
        """
        return cls.get_key(KeyType.IMAGE, window_id)
    
    @classmethod
    def get_mask_key(cls, window_id: Optional[str] = None) -> str:
        """
        Get NPZ key for mask array.
        
        Args:
            window_id: Window identifier. If None, returns single-window key.
                      If provided, returns multi-window key with window_id prefix.
        
        Returns:
            str: Key name for accessing mask in NPZ file
        
        Examples:
            >>> KeyManager.get_mask_key()
            'mask'
            >>> KeyManager.get_mask_key('window_1')
            'window_1_mask'
        """
        return cls.get_key(KeyType.MASK, window_id)
    
    @classmethod
    def get_key(cls, key_type: KeyType, window_id: Optional[str] = None) -> str:
        """
        Get NPZ key by KeyType enum and optional window ID.
        
        Args:
            key_type: KeyType enum member (IMAGE or MASK)
            window_id: Window identifier. If None, returns single-window key.
        
        Returns:
            str: Key name for accessing array in NPZ file
        
        Examples:
            >>> KeyManager.get_key(KeyType.IMAGE)
            'image'
            >>> KeyManager.get_key(KeyType.MASK, 'window_2')
            'window_2_mask'
        """
        if window_id is None:
            # Single window: use dispatcher map
            return cls._SINGLE_WINDOW_KEYS[key_type]
        
        # Multi window: window_id + separator + key_type value
        return f"{window_id}{cls.MULTI_WINDOW_SEPARATOR}{key_type.value}"
    
    @classmethod
    def is_single_window_image_key(cls, key: str) -> bool:
        """
        Check if key is for single-window image.
        
        Args:
            key: Key name to check
        
        Returns:
            bool: True if key is the single-window image key
        """
        return key == cls.SINGLE_WINDOW_IMAGE_KEY
    
    @classmethod
    def is_single_window_mask_key(cls, key: str) -> bool:
        """
        Check if key is for single-window mask.
        
        Args:
            key: Key name to check
        
        Returns:
            bool: True if key is the single-window mask key
        """
        return key == cls.SINGLE_WINDOW_MASK_KEY
    
    @classmethod
    def is_multi_window_image_key(cls, key: str) -> bool:
        """
        Check if key is for multi-window image.
        
        Args:
            key: Key name to check
        
        Returns:
            bool: True if key ends with image key type and is multi-window
        """
        suffix = f"{cls.MULTI_WINDOW_SEPARATOR}{KeyType.IMAGE.value}"
        return key.endswith(suffix) and key != cls.SINGLE_WINDOW_IMAGE_KEY
    
    @classmethod
    def is_multi_window_mask_key(cls, key: str) -> bool:
        """
        Check if key is for multi-window mask.
        
        Args:
            key: Key name to check
        
        Returns:
            bool: True if key ends with mask key type and is multi-window
        """
        suffix = f"{cls.MULTI_WINDOW_SEPARATOR}{KeyType.MASK.value}"
        return key.endswith(suffix) and key != cls.SINGLE_WINDOW_MASK_KEY
    
    @classmethod
    def extract_window_id(cls, key: str) -> Optional[str]:
        """
        Extract window ID from multi-window key.
        
        Args:
            key: Key name (e.g., 'window_1_image')
        
        Returns:
            str | None: Window ID if multi-window key, None if single-window key
        
        Examples:
            >>> KeyManager.extract_window_id('image')
            None
            >>> KeyManager.extract_window_id('window_1_image')
            'window_1'
        """
        if cls.is_single_window_image_key(key) or cls.is_single_window_mask_key(key):
            return None
        
        # Try both image and mask suffixes
        for key_type in KeyType:
            suffix = f"{cls.MULTI_WINDOW_SEPARATOR}{key_type.value}"
            if key.endswith(suffix):
                return key[:-len(suffix)]
        
        return None
