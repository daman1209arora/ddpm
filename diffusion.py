import torch
from torch import nn
import torch.functional as F
import matplotlib.pyplot as plt
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from datetime import datetime
import unet
from unet import UNetModel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

training_set = datasets.MNIST('../data', train=True, transform=transform, download=True)
validation_set = datasets.MNIST('../data', train=False, transform=transform, download=True)

training_loader = torch.utils.data.DataLoader(training_set, batch_size=32, shuffle=True, num_workers=1)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=32, shuffle=False, num_workers=1)

model = UNetModel(
    in_channels=1, 
    model_channels=64, 
    out_channels=1,
    num_res_blocks=1,
    attention_resolutions=(4,),
    channel_mult=(1, 2, 4)
)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

num_diffusion_steps = 1000
var_schedule = torch.linspace(1e-4, 2e-2, num_diffusion_steps)
cum_var_schedule = torch.cumprod(1-var_schedule, dim=0)

def train_one_epoch(epoch_index, tb_writer):
    global cum_var_schedule
    train_loss = []
    running_loss = 0
    last_loss = 0
    for i, data in enumerate(training_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        batch_size = inputs.shape[0]
        optimizer.zero_grad()
        
        timesteps = torch.randint(0, num_diffusion_steps, (batch_size, )).to(device)
        noise = torch.randn_like(inputs)
        noise = noise.to(device)
        cum_var_schedule_ = cum_var_schedule.to(device)
        cum_var = cum_var_schedule_[timesteps][:,None,None,None]
        noisy_images = torch.sqrt(cum_var)*inputs + torch.sqrt(1 - cum_var)*noise
        
        predicted_noise = model(noisy_images, timesteps)
        loss = nn.MSELoss()(noise, predicted_noise)
        
        loss.backward()
        optimizer.step()
        
        loss = loss.detach().cpu()
        running_loss += loss.item()
        train_loss.append(loss.item())
        
        if i % 100 == 99:
            last_loss = running_loss / 100 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            running_loss = 0

    return last_loss, train_loss

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 100
best_vloss = 1_000_000

train_loss, val_loss = [], []

for epoch in tqdm(range(EPOCHS), desc="Epochs"):
    print('EPOCH {}:'.format(epoch_number + 1))
    
    with torch.no_grad():
        model.train(False)
        samples = []
        noise = torch.randn(32, 1, 28, 28)
        X = noise.to(device)
        for i in range(num_diffusion_steps-1, -1, -1):
            timestep = torch.ones(X.shape[0])*i
            timestep = timestep.to(device)
            refined = model(X, timestep)
            beta = var_schedule[i]
            alpha = 1 - beta
            alpha_ = cum_var_schedule[i]
            X = (1 / torch.sqrt(alpha)) * (X - (beta / torch.sqrt(1 - alpha_))*refined + torch.sqrt(beta) * torch.randn_like(refined))
        X = X.detach().cpu().numpy().squeeze()
        print(X[0].mean(), X[1].mean())
        fig, ax = plt.subplots(4, 8)
        for i in range(X.shape[0]):
            ax[i//8][i%8].imshow(X[i])
        plt.savefig('images/{}_{}.png'.format(timestamp, epoch))
        
        
    model.train(True)
    avg_loss, tracked_loss = train_one_epoch(epoch_number, writer)
    train_loss.extend(tracked_loss)
    
    
    plt.plot(train_loss)
    plt.show()

    # Track best performance, and save the model's state
    '''
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}'.format(timestamp)
        torch.save(model.state_dict(), model_path)
    '''
    epoch_number += 1