from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class EncodedBytesResult:
    """
    Result of encoding operation containing PNG bytes and masks

    Encapsulates encoding results to avoid Dict[str, bytes] returns
    """
    images: Dict[str, bytes] = field(default_factory=dict)
    masks: Dict[str, Optional[bytes]] = field(default_factory=dict)

    def get_image(self, window_id: str = "window_1") -> Optional[bytes]:
        """Get image bytes for specific window"""
        return self.images.get(window_id)

    def get_mask(self, window_id: str = "window_1") -> Optional[bytes]:
        """Get mask bytes for specific window"""
        return self.masks.get(window_id)

    def add_window(self, window_id: str, image: bytes, mask: Optional[bytes] = None) -> None:
        """Add window encoding result"""
        self.images[window_id] = image
        self.masks[window_id] = mask

    def window_ids(self) -> list:
        """Get list of window IDs"""
        return list(self.images.keys())
