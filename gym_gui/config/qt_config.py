"""Centralized Qt configuration management.

This module provides a centralized way to manage Qt settings, ensuring that
configuration files are stored in the project's var/config/ directory instead
of polluting the user's home directory (~/.config/).
"""

from pathlib import Path
from qtpy import QtCore


class ProjectQtConfig:
    """Manages Qt settings in project var/ directory.
    
    This class centralizes Qt configuration management to prevent Qt from
    creating configuration files in the user's home directory. Instead,
    all configuration is stored in the project's var/config/ directory.
    
    Usage:
        # Get settings for a specific component
        settings = ProjectQtConfig.get_settings("GymGUI", "ControlPanelWidget")
        settings.setValue("key", "value")
        
        # List all configuration files
        configs = ProjectQtConfig.list_configs()
        
        # Reset all configuration (for testing)
        ProjectQtConfig.reset_all()
    """
    
    _CONFIG_DIR: Path | None = None
    
    @classmethod
    def set_config_dir(cls, path: Path) -> None:
        """Set custom config directory (primarily for testing).
        
        Args:
            path: Path to use as config directory
        """
        cls._CONFIG_DIR = path
    
    @classmethod
    def get_config_dir(cls) -> Path:
        """Get config directory, creating if needed.
        
        Returns:
            Path to the config directory (project_root/var/config)
        """
        if cls._CONFIG_DIR is None:
            # Default: project_root/var/config
            # __file__ is gym_gui/config/qt_config.py
            # parent.parent.parent gets us to project root
            project_root = Path(__file__).parent.parent.parent
            cls._CONFIG_DIR = project_root / "var" / "config"
        
        cls._CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        return cls._CONFIG_DIR
    
    @classmethod
    def get_settings(cls, org: str, app: str) -> QtCore.QSettings:
        """Get QSettings pointing to project var/ directory.
        
        Args:
            org: Organization name (e.g., "GymGUI")
            app: Application name (e.g., "ControlPanelWidget")
        
        Returns:
            QSettings configured to use INI file in var/config/
            
        Example:
            settings = ProjectQtConfig.get_settings("GymGUI", "ControlPanelWidget")
            settings.setValue("window_geometry", geometry)
            value = settings.value("window_geometry")
        """
        config_dir = cls.get_config_dir()
        path = config_dir / f"{org}_{app}.ini"
        return QtCore.QSettings(
            str(path),
            QtCore.QSettings.Format.IniFormat
        )
    
    @classmethod
    def reset_all(cls) -> None:
        """Clear all persisted configuration (for testing).
        
        This removes all configuration files from the var/config/ directory.
        Useful for testing to ensure clean state.
        """
        config_dir = cls.get_config_dir()
        if config_dir.exists():
            import shutil
            shutil.rmtree(config_dir)
            cls._CONFIG_DIR = None
    
    @classmethod
    def list_configs(cls) -> list[str]:
        """List all persisted config files.
        
        Returns:
            List of configuration file names (e.g., ["GymGUI_ControlPanelWidget.ini"])
        """
        config_dir = cls.get_config_dir()
        return [f.name for f in config_dir.glob("*.ini")]

