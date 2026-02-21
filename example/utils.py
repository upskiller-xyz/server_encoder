import sys
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from src.core.enums import FileFormat


def show_individual_channels(img_path:str)->None:
    img = np.array(Image.open("{}{}".format(img_path, FileFormat.PNG.value)))

    _, axes = plt.subplots(2, 2, figsize=(10, 10))
    channels = [('Red', img[:,:,0], 'Reds'), ('Green', img[:,:,1], 'Greens'), 
                ('Blue', img[:,:,2], 'Blues'), ('Alpha', img[:,:,3], 'gray')]

    for idx, (name, channel, cmap) in enumerate(channels):
        axes[idx // 2, idx % 2].imshow(channel, cmap=cmap)
        axes[idx // 2, idx % 2].set_title(name)
        axes[idx // 2, idx % 2].axis('off')

    plt.tight_layout()
    plt.show()