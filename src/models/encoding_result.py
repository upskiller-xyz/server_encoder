from typing import Dict, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class EncodingResult:
    """
    Result of encoding operation containing images and masks

    Encapsulates encoding results to avoid Dict[str, ...] returns
    """
    images: Dict[str, np.ndarray] = field(default_factory=dict)
    masks: Dict[str, Optional[np.ndarray]] = field(default_factory=dict)

    def get_image(self, window_id: str = "window_1") -> np.ndarray | None:
        """Get image for specific window"""
        return self.images.get(window_id)

    def get_mask(self, window_id: str = "window_1") -> Optional[np.ndarray]:
        """Get mask for specific window"""
        return self.masks.get(window_id)

    def add_window(self, window_id: str, image: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        """Add window encoding result"""
        self.images[window_id] = image
        self.masks[window_id] = mask

    def window_ids(self) -> list:
        """Get list of window IDs"""
        return list(self.images.keys())

    def is_single_window(self) -> bool:
        """Check if this is a single window result"""
        return len(self.images) == 1

    def get_first_image(self) -> np.ndarray:
        """Get first image (useful for single-window case)"""
        return next(iter(self.images.values()))

    def get_first_mask(self) -> Optional[np.ndarray]:
        """Get first mask (useful for single-window case)"""
        return next(iter(self.masks.values()))
