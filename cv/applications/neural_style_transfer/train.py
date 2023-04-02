import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import os


class VGG(nn.Module):
    def __init__(self):
        super().__init__()

        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_num) in self.chosen_features:
                features.append(x)

        return features
    


def load_image(image_name, transform):
    image = Image.open(image_name)
    image = transform(image).unsqueeze(0)
    return image



def train(content_img_path, style_img_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 1024

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    content_img = load_image(content_img_path, transform).to(device)
    style_img = load_image(style_img_path, transform).to(device)

    model = VGG().to(device)
    model.eval()

    # Randomly generated image, but usually copy from the orignal image
    # yield better result.
    # generated = torch.randn(content_img.shape, device=device, requires_grad=True)
    generated = content_img.clone().requires_grad_(True)

    # Hyperparameters
    total_steps = 6000
    learning_rate = 0.001
    alpha = 1
    beta = 0.01
    optimizer = optim.Adam([generated], lr=learning_rate)

    for step in range(total_steps):
        generated_features = model(generated)
        content_img_features = model(content_img)
        style_features = model(style_img)

        content_loss = 0
        style_loss = 0

        for gen_feature, content_feature, style_feature in zip(
            generated_features, content_img_features, style_features
        ):
            batch_size, channel, height, width = gen_feature.shape

            # content loss
            content_loss += torch.mean((gen_feature - content_feature) ** 2)

            # style loss: compute gram matrix
            G = gen_feature.view(channel, height*width).mm(
                gen_feature.view(channel, height*width).t()
            )
            A = style_feature.view(channel, height*width).mm(
                style_feature.view(channel, height*width).t()
            )
            style_loss += torch.mean((G - A) ** 2)

        total_loss = alpha * content_loss + beta * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"step: {step}, content loss: {content_loss:6.3f}, style loss: {style_loss:6.3f}, total loss: {total_loss:6.3f}", flush=True)
            save_image(generated, f"{output_path}/step_{step}.png")


if __name__ == "__main__":
    content_img_path = "tony.jpeg"

    for style_index in range(1, 9):
        print("Generating style:", style_index)
        style_img_path = f"styles/style{style_index}.jpg"
        output_path = f"output/tony_style_{style_index}"
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        train(content_img_path, style_img_path, output_path)

