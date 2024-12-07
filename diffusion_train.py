import os, math
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from datasets import ISICDataset  # Replace with the dataset you're using
from models import DeepDermClassifier  # Pretrained DeepDerm classifier
import torch.nn as nn
import torch.utils.data


def nonlinearity(x):
    ''' Also called the activation function. '''
    # swish
    return x*torch.sigmoid(x)
    # Swish is similar to GeLU. People tend to use this more than ReLU nowadays.

class Block(nn.Module):
    '''
    This implements a residual block.
    It has a similar structure to the residual block used in ResNets,
    but there are a few modern modifications:
     - Different order of applying weights, activations, and normalization.
     - Swish instead of ReLU activation.
     - GroupNorm instead of BatchNorm.
    We also need to add the conditional embedding.

    '''
    def __init__(self, in_channels, out_channels, emb_dim=256):
        '''
        in_channels: Number of image channels in input.
        out_channels: Number of image channels in output.
        emb_dim: Length of conditional embedding vector.
        '''
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(1, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        
        self.proj = nn.Linear(emb_dim, out_channels)

    def forward(self, x, t):
        '''
        h and x have dimension B x C x H x W,
        where B is batch size,
              C is channel size,
              H is height,
              W is width.
        t is the conditional embedding.
        t has dimension B x V,
        where V is the embedding dimension.
        '''
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        # Add conditioning to the hidden feature map h
        # (1) Linear projection of the conditional embedding t
        t_proj = self.proj(t)
        t_proj = nonlinearity(t_proj)

        # (3) Reshape for broadcasting across H and W dimensions
        # t_proj is reshaped to B x C x 1 x 1 so that it can be broadcasted
        t_proj = t_proj[:, :, None, None]
        
        # (3) Add the conditioning to h
        h = h + t_proj

        return h
    
class Down(nn.Module):
    ''' Downsampling block.'''
    def __init__(self, in_channels, out_channels):
        '''
        This block downsamples the feature map size by 2.
        in_channels: Number of image channels in input.
        out_channels: Number of image channels in output.
        '''
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = Block(in_channels, out_channels)

    def forward(self, x, t):
        ''' x is the feature maps; t is the conditional embeddings. '''
        x = self.pool(x) # The max pooling decreases feature map size by factor of 2
        x = self.conv(x, t)
        return x

class Up(nn.Module):
    ''' Upsampling block.'''
    def __init__(self, in_channels, out_channels):
        '''
        This block upsamples the feature map size by 2.
        in_channels: Number of image channels in input.
        out_channels: Number of image channels in output.
        '''
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = Block(in_channels, out_channels)

    def forward(self, x, skip_x, t):
        '''
        x is the feature maps;
        skip_x is the skipp connection feature maps;
        t is the conditional embeddings.
        '''
        x = self.up(x) # The upsampling increases the feature map size by factor of 2
        x = torch.cat([skip_x, x], dim=1) # concatentate skip connection
        x = self.conv(x, t)
        return x
    
class UNet(nn.Module):
    ''' UNet implementation of a denoising auto-encoder. '''
    def __init__(self, c_in=3, c_out=3, conditional=True, emb_dim=256):
        super().__init__()
        self.emb_dim = emb_dim
        self.inc = Block(c_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)

        self.bot1 = Block(256, 512)
        self.bot2 = Block(512, 512)
        self.bot3 = Block(512, 512)
        self.bot4 = Block(512, 256)

        self.up1 = Up(512, 128)
        self.up2 = Up(256, 64)
        self.up3 = Up(128, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        self.conditional = conditional
        if conditional:
            self.prob_projection = nn.Linear(1, emb_dim)  # Linear projection for scalar probability

    def temporal_encoding(self, timestep):
        '''
        Sinusoidal temporal encoding for timesteps.
        '''
        half_dim = self.emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=timestep.device)
        emb = timestep.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.emb_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    def unet_forward(self, x, t):
        '''
        Forward pass through UNet.
        '''
        x1 = self.inc(x, t)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)

        x4 = self.bot1(x4, t)
        x4 = self.bot2(x4, t)
        x4 = self.bot3(x4, t)
        x4 = self.bot4(x4, t)

        x = self.up1(x4, x3, t)
        x = self.up2(x, x2, t)
        x = self.up3(x, x1, t)
        output = self.outc(x)
        return output

    def forward(self, x, t, y=None):
        '''
        x: Image input
        t: Integer timestep
        y: Scalar probability
        '''
        temp_emb = self.temporal_encoding(t)

        if self.conditional and y is not None:
            y = y.view(-1, 1)  # Reshape y to [B, 1]
            prob_emb = self.prob_projection(y)  # Project probability to embedding space
            c = temp_emb + prob_emb
        else:
            c = temp_emb

        return self.unet_forward(x, c)


class Diffusion:
    '''
    Implements the Diffusion process,
    including both training and sampling.
    '''
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        ################## YOUR CODE HERE ##################
        # Here you should instantiate a 1D vector called self.beta,
        # which contains the \beta_t values
        # We use 1000 time steps, so t = 1:1000
        # \beta_1 = 1e-4
        # \beta_1000 = 0.02
        # The value of beta should increase linearly w.r.t. the value of t.
        #
        # Additionally, it may be helpful to pre-calculate the values of
        # \alpha_t and \bar{\alpha}_t here, since you'll use them often.

        ####################################################
        self.beta = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.alpha_bar_sqrt = torch.sqrt(self.alpha_bar)
        self.one_minus_alpha_bar_sqrt = torch.sqrt(1 - self.alpha_bar)

    def get_noisy_image(self, x_0, t):
        '''
        This function is only used for training.

        x_0: The input image. Dimensions: B x 3 x H x W
        t: A 1D vector of length B representing the desired timestep
          B is the batch size.
          H and W are the height and width of the input image.

        This function returns a *tuple of TWO tensors*:
            (x_t, epsilon)
            both have dimensions B x 3 x H x W
        '''
        ################## YOUR CODE HERE ##################
        # Calculate x_t from x_0 and t based on the equation you derived in problem 1.
        # Remember that \epsilon in the equation is noise drawn from
        # a standard normal distribution.
        # *** Return BOTH x_t and \epsilon as a tuple ***.

        ####################################################
        epsilon = torch.randn_like(x_0)
        
        alpha_bar_sqrt_t = self.alpha_bar_sqrt[t].view(-1, 1, 1, 1)
        one_minus_alpha_bar_sqrt_t = self.one_minus_alpha_bar_sqrt[t].view(-1, 1, 1, 1)
        x_t = alpha_bar_sqrt_t * x_0 + one_minus_alpha_bar_sqrt_t * epsilon
        return x_t, epsilon
        
        

    def sample(self, model, n, y=None):
        '''
        This function is used  to generate images.

        model: The denoising auto-encoder epsilon_{theta}
        n: The number of images you want to generate
        y: A 1D binary vector of size n indicating the
            desired gender for the generated face.
        '''
        model.eval()
        with torch.no_grad():
            ################## YOUR CODE HERE ##################
            # Write code for the sampling process here.
            # This process starts with x_T and progresses to x_0, T=1000
            # Reference *Algorithm 2* in "Denoising Diffusion Probabilistic Models" by Jonathan Ho et al.
            #
            # Start with x_T drawn from the standard normal distribution.
            # x_T has dimensions n x 3 x H x W.
            # H = W = 64 are the dimensions of the image for this assignment.
            #
            # Then for t = 1000 -> 1
            #     (1) Call the model to calculate \epsilon_{\theta}(x_t, t)
            #     (2) Use the formula from above to calculate \mu_{\theta} from \epsilon_{\theta}
            #     (3) Add zero-mean Gaussian noise with variance \beta_t to \mu_{\theta}
            #         this yields x_{t-1}
            #
            # Skip step (3) if t=1, because x_0 is the final image. It makes no sense to add noise to
            # the final product.

            ####################################################
            # Start with x_T drawn from standard normal distribution
            x_t = torch.randn((n, 3, self.img_size, self.img_size), device=self.device)
            
            # Ensure y matches the batch size `n`
            if y is not None:
                y = y[:n]  # Ensure `y` has the correct size
    
            for t in range(self.num_timesteps - 1, -1, -1):
                # Create a tensor with current timestep for model input
                t_tensor = torch.full((n,), t, device=self.device, dtype=torch.long)
    
                # (1) Call the model to predict epsilon
                if y is not None:
                    epsilon_theta = model(x_t, t_tensor, y)
                else:
                    epsilon_theta = model(x_t, t_tensor)
    
                # (2) Calculate mean (mu_theta) for the next step
                alpha_t = self.alpha[t]
                alpha_bar_t = self.alpha_bar[t]
                mean = (1 / torch.sqrt(alpha_t)) * (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * epsilon_theta)
    
                # (3) Add Gaussian noise if t > 1
                if t > 1:
                    noise = torch.randn_like(x_t)
                    x_t = mean + torch.sqrt(self.beta[t]) * noise
                else:
                    x_t = mean  # Final image without added noise
                    
        model.train()
        x_t = (x_t.clamp(-1, 1) + 1) / 2
        x_t = (x_t * 255).type(torch.uint8)
        return x_t


def main():
    # Hyperparameters
    num_epochs = 30
    batch_size = 4  # Small batch size per gradient step
    accumulate_steps = 8  # Number of steps for gradient accumulation
    effective_batch_size = batch_size * accumulate_steps  # Effective batch size
    save_path = 'diffusion_checkpoint.pth'
    save_interval = 100
    img_size = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.2)),  # Resize to 20% larger
        transforms.RandomCrop(img_size),        # Random crop to target size
        transforms.ColorJitter(0.2, 0, 0, 0),   # Apply brightness jitter
        transforms.ToTensor(),                  # Convert to tensor
        transforms.Normalize(mean=0.5, std=0.5) # Normalize to [-1, 1]
    ])
    dataset = ISICDataset(transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=effective_batch_size,  # Load effective batch size
        shuffle=True, 
        drop_last=True, 
        persistent_workers=True, 
        num_workers=2
    )

    # Models
    classifier = DeepDermClassifier().to(device)
    classifier.eval()
    model = UNet(c_in=3, c_out=3, conditional=True).to(device)
    diffusion = Diffusion(img_size=img_size, device=device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    # Loss function
    criterion = torch.nn.MSELoss()

    # Specify the positive class index
    positive_index = classifier.positive_index  # `1` for the positive class

    # Training loop
    writer = SummaryWriter()
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader)
        total_loss = 0
        for batch in pbar:
            images, _ = batch  # Assuming the dataset returns (image, label)
            images = images.to(device)

            # Split into smaller batches for gradient accumulation
            optimizer.zero_grad()
            for step in range(accumulate_steps):
                start = step * batch_size
                end = (step + 1) * batch_size
                images_step = images[start:end]  # Slice smaller batch

                # Get noisy images and noise
                timesteps = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)
                noisy_images, noise = diffusion.get_noisy_image(images_step, timesteps)

                # Get the probability of the positive class from DeepDermClassifier
                y_prob = classifier(images_step)[:, positive_index]

                # Predict noise using UNet
                predicted_noise = model(noisy_images, timesteps, y=y_prob)

                # Compute loss and accumulate gradients
                loss = criterion(predicted_noise, noise) / accumulate_steps
                loss.backward()
                total_loss += loss.item()

            # Update parameters after accumulating gradients
            optimizer.step()
            pbar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

        writer.add_scalar('Loss/train', total_loss / len(dataloader), epoch)

        # Save checkpoint every epoch
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch}, save_path)

        # Backup every `save_interval` epochs
        if epoch % save_interval == 0:
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch}, f"{save_path}.{epoch}")

    writer.close()

if __name__ == "__main__":
    main()

