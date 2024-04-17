import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAEModel, self).__init__()
        
        # Define the encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Define the decoder
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h2 = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h2))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# You can then instantiate the model with the required dimensions
# For example: VAE with 784-dimensional input (e.g., flattened 28x28 images from MNIST), 400 hidden units, and 20 latent variables
input_dim = 784
hidden_dim = 400
latent_dim = 20

model = VAEModel(input_dim, hidden_dim, latent_dim)

# Example forward pass with random data
data = torch.randn(64, input_dim) # Batch size of 64
reconstructed, mu, logvar = model(data)

# You may want to add additional methods for loss computation, training steps, etc.
