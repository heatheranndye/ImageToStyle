from pathlib import Path
import torch

import pytest

from ImageToStyle.ImageToStyle.photo_mod.styletransfer import (
    image_loader,
    VGG,
    device,
    calculate_style_loss,
    calculate_content_loss,
    image_trainer,
)

DATAPATH = Path(__file__).parent.parent / "ImageToStyle/photo_mod/data"

CONTENT_PATH = DATAPATH / "squareflower.jpg"
STYLE_PATH = DATAPATH / "squarebox.jpg"


@pytest.fixture
def the_model():
    return VGG().to(device).eval()


@pytest.fixture
def sample_image():
    return image_loader(CONTENT_PATH)


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


def test_style_loss(the_model, sample_image):
    # test style loss
    sample_feature = the_model(sample_image)
    loss = calculate_style_loss(sample_feature, sample_feature)
    assert type(loss) == torch.Tensor


def test_content_loss(the_model, sample_image):
    # test content loss
    sample_feature = the_model(sample_image)
    loss = calculate_content_loss(sample_feature, sample_feature)
    assert type(loss) == torch.Tensor


def test_trainer():
    # test the trainer
    image_trainer()
