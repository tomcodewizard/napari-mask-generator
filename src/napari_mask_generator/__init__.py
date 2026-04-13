# src/napari_mask_generator/__init__.py
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("napari-mask-generator")
except PackageNotFoundError:
    __version__ = "uninstalled"


def activate(context):
    pass