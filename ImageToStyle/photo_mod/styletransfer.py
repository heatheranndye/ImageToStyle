import pathlib
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from torchvision.utils import save_image
import torch.optim as optim
import torch.nn.functional as F

DATAPATH = pathlib.Path(__file__).parent / "data"

CONTENT_PATH = DATAPATH / "squareflower.jpg"
STYLE_PATH = DATAPATH / "YellowFlowerWaterPixelTrim.jpg"
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
model = models.vgg19(weights="IMAGENET1K_V1").features
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
    """The model uses the pre-trained VGG19 model.  We
     The forward method returns the results of applying
     the layers in req_features as described in
    arxiv.org/pdf/1508.06576.pdf.
     The neural network is truncated at layer 28.
     Args:
         nn (_type_): Base class for all neural network modules.
    """

    def __init__(self):
        super(VGG, self).__init__()
        self.req_features = ["0", "5", "10", "19", "28"]
        self.model = models.vgg19(weights="IMAGENET1K_V1").features[:29]

    def forward(self, output_layer) -> list:
        """Returns the output of selected layers of
        the neural network.

        Args:
            layer output: recursive variable for the layer
            outputs. The first layer output is the torch
            tensor constructed from an image

        Returns:
            List: A list of 5 torch tensors.
        """
        features = []
        # Iterate over all the layers of the mode
        for layer_num, layer in enumerate(self.model):
            # activation of the layer will stored in x
            output_layer = layer(output_layer)
            # appending the activation of the selected
            #  layers and return the feature array
            if str(layer_num) in self.req_features[0:]:
                features.append(output_layer)
        return features


def calc_style_loss(
    gen_feat,
    style_feat,
):
    # Calculating the gram matrix for the style and the generated image
    style_l = 0
    for x in range(0, 5, 1):
        # print(x)
        batch, channel, height, width = gen_feat[x].shape
        genitem = gen_feat[x]
        styleitem = style_feat[x]
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
    gen_feat,
    cont_feat,
    style_feat,
):
    style_loss = content_loss = 0
    # extracting the dimensions from the generated image
    content_loss += calc_content_loss(gen_feat, cont_feat)
    style_loss += calc_style_loss(gen_feat, style_feat)

    # calculating the total loss of e th epoch
    total_loss = alpha * content_loss + beta * style_loss
    return total_loss


def image_trainer() -> Image:
    original_image = image_loader(DATAPATH / "squareflower.jpg")
    style_image = image_loader(DATAPATH / "YellowFlowerWaterPixelTrim.jpg")
    image_name = "gen.jpg"
    model = VGG().to(device).eval()

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


if __name__ == "__main__":
    image_trainer()
