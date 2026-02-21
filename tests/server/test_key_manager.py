"""Tests for NPZ key manager"""
import pytest
from src.server.key_manager import KeyManager, KeyType


class TestKeyManagerGetImageKey:
    """Test image key generation"""

    def test_get_single_window_image_key(self):
        """Test getting single-window image key"""
        key = KeyManager.get_image_key()
        assert key == 'image'

    def test_get_single_window_image_key_with_none(self):
        """Test getting single-window image key with explicit None"""
        key = KeyManager.get_image_key(None)
        assert key == 'image'

    def test_get_multi_window_image_key(self):
        """Test getting multi-window image key"""
        key = KeyManager.get_image_key('window_1')
        assert key == 'window_1_image'

    def test_get_multi_window_image_key_different_ids(self):
        """Test multi-window image keys with different window IDs"""
        test_cases = [
            ('window1', 'window1_image'),
            ('w1', 'w1_image'),
            ('left_window', 'left_window_image'),
            ('da_window_2', 'da_window_2_image'),
        ]
        
        for window_id, expected in test_cases:
            key = KeyManager.get_image_key(window_id)
            assert key == expected


class TestKeyManagerGetMaskKey:
    """Test mask key generation"""

    def test_get_single_window_mask_key(self):
        """Test getting single-window mask key"""
        key = KeyManager.get_mask_key()
        assert key == 'mask'

    def test_get_single_window_mask_key_with_none(self):
        """Test getting single-window mask key with explicit None"""
        key = KeyManager.get_mask_key(None)
        assert key == 'mask'

    def test_get_multi_window_mask_key(self):
        """Test getting multi-window mask key"""
        key = KeyManager.get_mask_key('window_1')
        assert key == 'window_1_mask'

    def test_get_multi_window_mask_key_different_ids(self):
        """Test multi-window mask keys with different window IDs"""
        test_cases = [
            ('window1', 'window1_mask'),
            ('w1', 'w1_mask'),
            ('left_window', 'left_window_mask'),
            ('da_window_2', 'da_window_2_mask'),
        ]
        
        for window_id, expected in test_cases:
            key = KeyManager.get_mask_key(window_id)
            assert key == expected


class TestKeyManagerGetKey:
    """Test get_key with KeyType enum and window ID"""

    def test_get_single_window_image_by_type(self):
        """Test getting single-window image using get_key"""
        key = KeyManager.get_key(KeyType.IMAGE)
        assert key == 'image'

    def test_get_single_window_mask_by_type(self):
        """Test getting single-window mask using get_key"""
        key = KeyManager.get_key(KeyType.MASK)
        assert key == 'mask'

    def test_get_multi_window_image_by_type(self):
        """Test getting multi-window image using get_key"""
        key = KeyManager.get_key(KeyType.IMAGE, 'window_1')
        assert key == 'window_1_image'

    def test_get_multi_window_mask_by_type(self):
        """Test getting multi-window mask using get_key"""
        key = KeyManager.get_key(KeyType.MASK, 'window_2')
        assert key == 'window_2_mask'

    def test_get_key_with_all_key_types(self):
        """Test get_key with all KeyType enum members"""
        for key_type in KeyType:
            single_key = KeyManager.get_key(key_type)
            multi_key = KeyManager.get_key(key_type, 'window_1')
            
            # Single window should be just the key type value
            assert single_key == key_type.value
            
            # Multi window should be window_id + separator + key type value
            assert multi_key == f"window_1_{key_type.value}"


class TestKeyManagerCheckers:
    """Test key type checker methods"""

    def test_is_single_window_image_key(self):
        """Test checking if key is single-window image"""
        assert KeyManager.is_single_window_image_key('image') is True
        assert KeyManager.is_single_window_image_key('mask') is False
        assert KeyManager.is_single_window_image_key('window_1_image') is False

    def test_is_single_window_mask_key(self):
        """Test checking if key is single-window mask"""
        assert KeyManager.is_single_window_mask_key('mask') is True
        assert KeyManager.is_single_window_mask_key('image') is False
        assert KeyManager.is_single_window_mask_key('window_1_mask') is False

    def test_is_multi_window_image_key(self):
        """Test checking if key is multi-window image"""
        assert KeyManager.is_multi_window_image_key('window_1_image') is True
        assert KeyManager.is_multi_window_image_key('w1_image') is True
        assert KeyManager.is_multi_window_image_key('image') is False
        assert KeyManager.is_multi_window_image_key('window_1_mask') is False
        assert KeyManager.is_multi_window_image_key('mask') is False

    def test_is_multi_window_mask_key(self):
        """Test checking if key is multi-window mask"""
        assert KeyManager.is_multi_window_mask_key('window_1_mask') is True
        assert KeyManager.is_multi_window_mask_key('w1_mask') is True
        assert KeyManager.is_multi_window_mask_key('mask') is False
        assert KeyManager.is_multi_window_mask_key('window_1_image') is False
        assert KeyManager.is_multi_window_mask_key('image') is False

    def test_checker_consistency(self):
        """Test that checkers are mutually exclusive and consistent"""
        single_image = KeyManager.get_image_key()
        single_mask = KeyManager.get_mask_key()
        multi_image = KeyManager.get_image_key('window_1')
        multi_mask = KeyManager.get_mask_key('window_2')

        # Single image
        assert KeyManager.is_single_window_image_key(single_image)
        assert not KeyManager.is_single_window_mask_key(single_image)
        assert not KeyManager.is_multi_window_image_key(single_image)
        assert not KeyManager.is_multi_window_mask_key(single_image)

        # Single mask
        assert KeyManager.is_single_window_mask_key(single_mask)
        assert not KeyManager.is_single_window_image_key(single_mask)
        assert not KeyManager.is_multi_window_image_key(single_mask)
        assert not KeyManager.is_multi_window_mask_key(single_mask)

        # Multi image
        assert KeyManager.is_multi_window_image_key(multi_image)
        assert not KeyManager.is_single_window_image_key(multi_image)
        assert not KeyManager.is_single_window_mask_key(multi_image)
        assert not KeyManager.is_multi_window_mask_key(multi_image)

        # Multi mask
        assert KeyManager.is_multi_window_mask_key(multi_mask)
        assert not KeyManager.is_single_window_image_key(multi_mask)
        assert not KeyManager.is_single_window_mask_key(multi_mask)
        assert not KeyManager.is_multi_window_image_key(multi_mask)


class TestKeyManagerExtractWindowId:
    """Test window ID extraction from keys"""

    def test_extract_window_id_from_single_image(self):
        """Test extracting window ID from single-window image key"""
        window_id = KeyManager.extract_window_id('image')
        assert window_id is None

    def test_extract_window_id_from_single_mask(self):
        """Test extracting window ID from single-window mask key"""
        window_id = KeyManager.extract_window_id('mask')
        assert window_id is None

    def test_extract_window_id_from_multi_image(self):
        """Test extracting window ID from multi-window image key"""
        window_id = KeyManager.extract_window_id('window_1_image')
        assert window_id == 'window_1'

    def test_extract_window_id_from_multi_mask(self):
        """Test extracting window ID from multi-window mask key"""
        window_id = KeyManager.extract_window_id('window_2_mask')
        assert window_id == 'window_2'

    def test_extract_window_id_different_ids(self):
        """Test extracting different window IDs"""
        test_cases = [
            ('w1_image', 'w1'),
            ('left_window_image', 'left_window'),
            ('da_window_2_mask', 'da_window_2'),
            ('window1_image', 'window1'),
        ]
        
        for key, expected_id in test_cases:
            extracted_id = KeyManager.extract_window_id(key)
            assert extracted_id == expected_id

    def test_extract_and_recreate_key(self):
        """Test that extracted window ID can recreate the original key"""
        original_keys = [
            KeyManager.get_image_key('window_1'),
            KeyManager.get_mask_key('window_2'),
            KeyManager.get_image_key('left'),
        ]
        
        for original_key in original_keys:
            window_id = KeyManager.extract_window_id(original_key)
            
            # Determine key type from original
            if original_key.endswith('_image'):
                recreated_key = KeyManager.get_image_key(window_id)
            else:
                recreated_key = KeyManager.get_mask_key(window_id)
            
            assert recreated_key == original_key


class TestKeyManagerConstantsAndPatterns:
    """Test key name constants and patterns"""

    def test_single_window_constants(self):
        """Test that single window constants are as expected"""
        assert KeyManager.SINGLE_WINDOW_IMAGE_KEY == 'image'
        assert KeyManager.SINGLE_WINDOW_MASK_KEY == 'mask'

    def test_multi_window_separator(self):
        """Test that multi-window separator is as expected"""
        assert KeyManager.MULTI_WINDOW_SEPARATOR == '_'

    def test_key_type_enum_values(self):
        """Test that KeyType enum has correct values"""
        assert KeyType.IMAGE.value == 'image'
        assert KeyType.MASK.value == 'mask'

    def test_backward_compatibility(self):
        """Test that keys maintain backward compatibility with server_lux"""
        # Single window should use old key names
        assert KeyManager.get_image_key() == 'image'
        assert KeyManager.get_mask_key() == 'mask'
        
        # Multi-window should use window ID prefix with separator
        assert KeyManager.get_image_key('window_1') == 'window_1_image'
        assert KeyManager.get_mask_key('window_1') == 'window_1_mask'
