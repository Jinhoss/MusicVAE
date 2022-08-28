import torch
import torch.optim as optim
from tqdm import tqdm
from time import time
from cal_function import accuracy, kl_annealing, vae_loss


def train(args, train_loader, val_loader, model, optimizer):
    history = {}
    history['train_loss'] = []
    history['train_acc'] = []
    history['val_loss'] = []
    history['val_acc'] = []
    loss_fn = vae_loss
    epochs = args.epochs
    device = args.device
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)   
    for i in range(1, epochs+1):
        start_time = time()
        train_loss = 0
        train_acc = 0     
        val_loss = 0
        val_acc = 0
        
        model.train()
        print(f'start train {i}')
        for batch_idx, x_train in tqdm(enumerate(train_loader)):
            x_train = x_train.to(device)         
            optimizer.zero_grad()     
            x_train_prob, x_train_mu, x_train_std, x_train_label = model(x_train)
            beta = kl_annealing(i, 0, 0.2)
            loss = loss_fn(x_train_prob, x_train, x_train_mu, x_train_std, beta)
            loss.backward()
            optimizer.step()          
            train_loss += loss.item()
            train_acc += accuracy(x_train, x_train_label).item()
            
            scheduler.step()
        
        train_loss = train_loss / (batch_idx + 1)
        train_acc = train_acc / (batch_idx + 1)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        model.eval()
        with torch.no_grad():
            for batch_idx, x_val in tqdm(enumerate(val_loader)):
                x_val = x_val.to(device)
                x_val_prob, x_val_mu, x_val_std, x_val_label = model(x_val)
                loss = loss_fn(x_val_prob, x_val, x_val_mu, x_val_std)
                
                val_loss += loss.item()
                val_acc += accuracy(x_val, x_val_label).item()
                
        val_loss = val_loss / (batch_idx + 1)
        val_acc = val_acc / (batch_idx + 1)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print('Epoch %d (%0.2f sec) - train_loss: %0.3f, train_acc: %0.3f, val_loss: %0.3f, val_acc: %0.3f, lr: %0.6f' % \
             (i, time()-start_time, train_loss, train_acc, val_loss, val_acc, scheduler.get_last_lr()[0]))

    torch.save({
        "model": model.state_dict()
    }, args.model_path
)
        
    return history