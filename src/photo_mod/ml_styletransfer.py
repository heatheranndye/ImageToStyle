import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
from torchvision.utils import save_image
import torch.optim as optim
import pathlib


DATAPATH = pathlib.Path(__file__).parent / "data"

model = models.vgg19(weights="IMAGENET1K_V1").features
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")


def image_retrieval(id_number: int) -> Image:
    """Retrieve an image from the data based based on an id number

    Args:
        id_number (int): image identification number

    Returns:
        Image:
    """
    image_num = "flower" + str(id_number) + ".jpg"
    LOCALPATH = DATAPATH / image_num
    image = Image.open(LOCALPATH).convert("RGB")
    return image


def texture_retrieval(id_number: int) -> Image:
    """Retrieve an image from the data based based on an id number

    Args:
        id_number (int): image identification number

    Returns:
        Image:
    """
    image_num = "texture" + str(id_number) + ".jpg"
    LOCALPATH = DATAPATH / image_num
    image = Image.open(LOCALPATH).convert("RGB")
    return image


def image_loader(path):
    image = Image.open(path)
    loader = transforms.Compose(
        [
            transforms.Resize(
                (
                    512,
                    512,
                )
            ),
            transforms.ToTensor(),
        ]
    )
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


# Defining a class that for the model


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
            if str(layer_num) in self.req_features:
                features.append(x)

        return features


def calc_content_loss(gen_feat, orig_feat):
    # calculating the content loss of each layer
    #  by calculating the MSE between the content and
    #  generated features and adding it to content loss
    content_l = torch.mean((gen_feat - orig_feat) ** 2)
    return content_l


def calc_style_loss(gen, style):
    # Calculating the gram matrix for the style and the generated image
    batch_size, channel, height, width = gen.shape

    G = torch.mm(
        gen.view(
            channel,
            height * width,
        ),
        gen.view(channel, height * width).t(),
    )
    A = torch.mm(
        style.view(
            channel,
            height * width,
        ),
        style.view(channel, height * width).t(),
    )

    # Calcultating the style loss of each layer by
    # calculating the MSE between the gram matrix of
    # the style image and the generated image and adding it to style loss
    style_l = torch.mean((G - A) ** 2)
    return style_l


def calculate_loss(
    gen_features,
    orig_feautes,
    style_featues,
):
    style_loss = content_loss = 0
    for gen, cont, style in zip(
        gen_features,
        orig_feautes,
        style_featues,
    ):
        # extracting the dimensions from the generated image
        content_loss += calc_content_loss(gen, cont)
        style_loss += calc_style_loss(gen, style)

    # calculating the total loss of e th epoch
    total_loss = alpha * content_loss + beta * style_loss
    return total_loss


# Load the model to the GPU
model = VGG().to(device).eval()

# initialize the paramerters required for fitting the model
epoch = 1000
lr = 0.04
alpha = 10
beta = 50


def image_trainer(label: int) -> Image:
    original_image = image_loader(DATAPATH / "flower3.jpg")
    style_image = image_loader(DATAPATH / "texture4.jpg")
    image_name = "gen" + str(label) + ".jpg"
    generated_image = original_image.clone().requires_grad_(True)

    # using adam optimizer and it will update
    #  the generated image not the model parameter
    optimizer = optim.Adam([generated_image], lr=lr)

    # iterating for 1000 times
    for e in range(100):
        # extracting the features of generated, content
        # and the original required for calculating the loss
        gen_features = model(generated_image)
        orig_feautures = model(original_image)
        style_features = model(style_image)

        # iterating over the activation of each layer
        # and calculate the loss and add it to the content and the style loss
        total_loss = calculate_loss(
            gen_features,
            orig_feautures,
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
