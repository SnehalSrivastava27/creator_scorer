# --- 2. UNIFIED BETA-VAE ARCHITECTURE ---
class BetaVAE(nn.Module):
    def __init__(self):
        super(BetaVAE, self).__init__()
        
        # Encoder Blocks
        self.enc1 = nn.Sequential(nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU())
        self.enc4 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU())
        
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(256*4*4, LATENT_DIM)
        self.fc_var = nn.Linear(256*4*4, LATENT_DIM)
        
        # Decoder
        self.decoder_input = nn.Linear(LATENT_DIM, 256*4*4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid() 
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        encoded = self.flatten(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_var(encoded)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder_input(z).view(-1, 256, 4, 4)
        reconstruction = self.decoder(decoded)
        return reconstruction, mu, logvar

    def extract_features(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        return f1, f2, f3, f4