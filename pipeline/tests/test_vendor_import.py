from pathlib import Path


def test_vendor_module_importable():
    """vendor.openmontage module should be importable."""
    from vendor.openmontage.tools.graphics.image_selector import ImageSelector

    assert ImageSelector is not None


def test_vendor_has_init_files():
    """vendor package directories should contain __init__.py files."""
    project_root = Path(__file__).resolve().parent.parent.parent

    assert (project_root / "vendor" / "__init__.py").exists()
    assert (project_root / "vendor" / "openmontage" / "__init__.py").exists()
