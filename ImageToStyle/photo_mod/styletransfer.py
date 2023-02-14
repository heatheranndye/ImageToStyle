import pathlib
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from torchvision.utils import save_image
import torch.optim as optim
import torch.nn.functional as F

DATAPATH = pathlib.Path.cwd() / "data"

CONTENT_PATH = DATAPATH / "squareflower.jpg"
STYLE_PATH = DATAPATH / "YellowFlowerWaterPixelTrim.jpg"
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
alpha = 1000
beta = 50


def image_loader(path):
    """Load the image file. Change the pixels to
    256 x 256 and then change to a Torch Tensor.

    Args:
        path (Image Path): Image Path
    """
    image = Image.open(path)
    loader = transforms.Compose(
        [
            transforms.Resize(
                (
                    256,
                    256,
                )
            ),
            transforms.ToTensor(),
        ]
    )
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.req_features = ["0", "5", "10", "19", "28"]
        # Since we need only the 5 layers in the model so we
        #  will be dropping all the rest layers from the features of the model
        self.model = models.vgg19(weights="IMAGENET1K_V1").features[
            :29
        ]  # model will contain the first 29 layers

    # x holds the input tensor(image) that will be feeded to each layer
    def forward(self, x):
        # initialize an array that wil hold the
        # activations from the chosen layers
        features = []
        # Iterate over all the layers of the mode
        for layer_num, layer in enumerate(self.model):
            # activation of the layer will stored in x
            x = layer(x)
            # appending the activation of the selected
            #  layers and return the feature array
            if str(layer_num) in self.req_features[0:]:
                features.append(x)

        return features


def calc_style_loss(gen, style):
    # Calculating the gram matrix for the style and the generated image
    style_l = 0
    for x in range(0, 5, 1):
        # print(x)
        batch, channel, height, width = gen[x].shape
        genitem = gen[x]
        styleitem = style[x]
        G = torch.mm(
            genitem.view(
                channel,
                height * width,
            ),
            genitem.view(channel, height * width).t(),
        )
        A = torch.mm(
            styleitem.view(
                channel,
                height * width,
            ),
            styleitem.view(channel, height * width).t(),
        )

        # Calcultating the style loss of each layer by
        # calculating the MSE between the gram matrix of
        # the style image and the generated image and adding it to style loss
        style_l += F.mse_loss(G, A)
    return style_l


def calc_content_loss(gen_feat, orig_feat):
    # calculating the content loss of each layer
    #  by calculating the MSE between the content and
    #  generated features and adding it to content loss
    content_l = 0
    x = 4
    # for x in range(len(gen_feat)):
    content_l += F.mse_loss(gen_feat[x], orig_feat[x])
    return content_l


def calculate_loss(
    gen,
    cont,
    style,
):
    style_loss = content_loss = 0
    # extracting the dimensions from the generated image
    content_loss += calc_content_loss(gen, cont)
    style_loss += calc_style_loss(gen, style)

    # calculating the total loss of e th epoch
    total_loss = alpha * content_loss + beta * style_loss
    return total_loss


def image_trainer() -> Image:
    original_image = image_loader(DATAPATH / "squareflower.jpg")
    style_image = image_loader(DATAPATH / "YellowFlowerWaterPixelTrim.jpg")
    image_name = "gen5.jpg"
    model = model = models.vgg19(weights="IMAGENET1K_V1").features
    lr = 0.04

    generated_image = original_image.clone().requires_grad_(True)
    model.requires_grad_(False)
    # using adam optimizer and it will update
    #  the generated image not the model parameter
    optimizer = optim.Adam([generated_image], lr=lr)

    # iterating for 1000 times
    for e in range(100):
        # extracting the features of generated, content
        # and the original required for calculating the loss
        gen_features = model(generated_image)
        orig_features = model(original_image)
        style_features = model(style_image)

        # iterating over the activation of each layer
        # and calculate the loss and add it to the content and the style loss
        total_loss = calculate_loss(
            gen_features,
            orig_features,
            style_features,
        )
        # optimize the pixel values of the generated
        # image and backpropagate the loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        # print the image and save it after each 100 epoch
        if e / 100:
            print(total_loss)

            save_image(generated_image, image_name)
