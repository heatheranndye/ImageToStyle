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
STYLE_PATH = DATAPATH / "picasso.jpg"
SAMPLE_PATH = DATAPATH / "samplegenpicasso.jpg"

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
model = models.vgg19(weights="IMAGENET1K_V1").features
alpha = 1000
beta = 50

style_layers_default = [1, 2, 3, 4, 5]
content_layers_default = [5]


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
        for layer_num, layer in enumerate(self.model):
            output_layer = layer(output_layer)
            # appending the activation of the selected
            #  layers and return the feature array
            if str(layer_num) in self.req_features[0:]:
                features.append(output_layer)
        return features


def calculate_style_loss(
    gen_feat,
    style_feat,
    style_layers=style_layers_default,
) -> torch.Tensor:
    """Sets up the Gram matrix for the generated
    feature layers and the style feature layers.
    Then compute the MSE.

    Args:
        gen_feat(Torch Tensors): list of Torch tensors
        style_feat (Torch Tensors): list of torch tensors
        style_layers (Torch Tensors, optional): list of
        integers. Defaults to style_layers_default.

    Returns:
        torch.Tensor: Style loss value
    """
    style_loss = 0
    for value in style_layers:
        counter = value - 1
        # print(f"{counter}")
        batch, channel, height, width = gen_feat[counter].shape
        genitem = gen_feat[counter]
        styleitem = style_feat[counter]
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
        style_loss += F.mse_loss(G, A)
    return style_loss


def calculate_content_loss(
    content_feat,
    gen_feat,
    content_layers=content_layers_default,
) -> torch.Tensor:
    """The MSE is calculated for list of features.

    Args:
        gen_feat (list): list of torch Tensors
        content_feat (list): list of torch Tensors
        content_layers (list, optional):layer list.
        Defaults to content_layers_default.

    Returns:
        _type_: _description_
    """
    content_loss = 0
    for value in content_layers:
        counter = value - 1
        # print(f"{counter}")
        content_loss += F.mse_loss(gen_feat[counter], content_feat[counter])
    return content_loss


def calculate_loss(
    content_feat,
    gen_feat,
    style_feat,
    content_layers=content_layers_default,
    style_layers=style_layers_default,
):
    """Sums the content loss and style loss.

    Args:
        gen_feat (list): _description_
        content_feat (list): _description_
        style_feat (list): _description_
        content_layers (list, optional): _description_.
        Defaults to content_layers_default.
        style_layers (list, optional): _description_.
        Defaults to style_layers_default.

    Returns:
        _type_: _description_
    """
    style_loss = content_loss = 0

    content_loss += calculate_content_loss(
        content_feat,
        gen_feat,
        content_layers,
    )
    style_loss += calculate_style_loss(
        gen_feat,
        style_feat,
        style_layers,
    )

    # calculating the total loss of e th epoch
    total_loss = alpha * content_loss + beta * style_loss
    return total_loss


def image_trainer(
    content_path=CONTENT_PATH,
    style_path=STYLE_PATH,
    alpha=alpha,
    beta=beta,
    content_layers=content_layers_default,
    style_layers=style_layers_default,
) -> Image:
    """Trains the generated image, which is
    initially a copy of the content image.

    Args:
        content_path (Path, optional): content image.
         Defaults to CONTENT_PATH.
        style_path (Path, optional): style image.
        Defaults to STYLE_PATH.
        content_layer (list, optional): content layers.
        Defaults to content_layers_default.
        style_layer (list, optional): style layers.
        Defaults to style_layers_default.

    Returns:
        Image: _description_
    """
    content_image = image_loader(content_path)
    style_image = image_loader(style_path)
    model = VGG().to(device).eval()

    lr = 0.04

    generated_image = content_image.clone().requires_grad_(True)
    model.requires_grad_(False)
    optimizer = optim.Adam([generated_image], lr=lr)
    content_features = model(content_image)
    style_features = model(style_image)

    for e in range(100):
        gen_features = model(generated_image)

        total_loss = calculate_loss(
            content_features,
            gen_features,
            style_features,
            style_layers=style_layers,
            content_layers=content_layers,
        )
        # optimize the pixel values of the generated
        # image and backpropagate the loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        # print the image and save it after each 100 epoch
        # if e / 100:
        # print(f"e: {e}, loss:{total_loss}")

    save_image(generated_image, "gen.jpg")
    image = Image.open("gen.jpg")
    return image


if __name__ == "__main__":
    image_test = image_trainer()
    save_image(image_test, "gen22.jpg")
