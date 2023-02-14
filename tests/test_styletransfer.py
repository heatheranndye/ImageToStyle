from pathlib import Path


import pytest

from ImageToStyle.ImageToStyle.photo_mod.styletransfer import (
    image_loader,
    VGG,
    device,
)

DATAPATH = Path(__file__).parent.parent / "ImageToStyle/photo_mod/data"

CONTENT_PATH = DATAPATH / "squareflower.jpg"
STYLE_PATH = DATAPATH / "squarebox.jpg"


@pytest.fixture
def the_model():
    return VGG().to(device).eval()


def test_image_loader():
    # test to ensure loader functions
    image_loader(STYLE_PATH)


def test_size_image_loader():
    # test to ensure images have same torch dimensions
    # this is important for the GANS to function correctly.
    image1 = image_loader(STYLE_PATH)
    image2 = image_loader(CONTENT_PATH)
    image1.shape == image2.shape


def test_class_VGG(the_model):
    # test the model
    image = image_loader(CONTENT_PATH)
    the_model(image)


def test_the_features(the_model):
    # test the model
    image = image_loader(CONTENT_PATH)
    image_features = the_model(image)
    assert len(image_features) == 5
