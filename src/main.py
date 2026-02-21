import os
import sys
from pathlib import Path

# Disable GPU/CUDA to prevent bus errors on WSL2
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.server.application import ServerApplication
from src.server.launcher import ServerLauncher


def main() -> None:
    """Main entry point"""
    launcher = ServerLauncher()
    application = launcher.create_application()
    port = int(os.getenv("PORT", 8082))
    launcher.run_server(application, port=port, debug=True)


# Create app instance for gunicorn only when needed
# Don't create at module import time to avoid bus errors
def create_app():
    """
    Factory function for creating the Flask app (for gunicorn).
    
    Returns:
        Flask: The Flask application instance
    """
    _application = ServerApplication()
    return _application.app


# Only create app instance if not running as main (i.e., when imported by gunicorn)
if __name__ != "__main__":
    app = create_app()
else:
    # Running as main script
    main()