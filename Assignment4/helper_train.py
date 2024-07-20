import time
import torch
import torch.nn.functional as F
import torchvision
import torch.autograd

def train_gan_v1(num_epochs, model, optimizer_gen, optimizer_discr, 
                 latent_dim, device, train_loader, loss_fn=None,
                 logging_interval=100, 
                 save_model=None):
    
    log_dict = {'train_generator_loss_per_batch': [],
            'train_discriminator_loss_per_batch': [],
            'train_discriminator_real_acc_per_batch': [],
            'train_discriminator_fake_acc_per_batch': [],
            'images_from_noise_per_epoch': []}
    
    
    if loss_fn is None :
        loss_fn = F.binary_cross_entropy_with_logits
        
     
     
    fixed_noise  = torch.randn(64,latent_dim,1,1,device = device ) # format NCHW    
    start_time  = time.time()
    for epoch in range(num_epochs):
        model.train()
        for batch_idx,(features,_) in enumerate(train_loader):
            batch_size = features.size(0)
                  
            # real images
            real_images = features.to(device)
            real_labels = torch.ones(batch_size,device=device)          # label = 1
            
            
            # generated (fake) images
            noise = torch.randn(batch_size , latent_dim , 1, 1 , device = device)    #format NCHW
            fake_images = model.generator_forward(noise)
            fake_labels  =  torch.zeros(batch_size , device =  device ) # fake label = 0
            flipped_fake_image = real_labels # here fake label =1 
            
            
            
            #-------------------------------
            
            # Train Discriminator 
            
            #-------------------------------
            
            
            optimizer_discr.zero_grad()
            
            # get discreminator loss on real images
            
            discr_pred_real = model.discriminator_forward(real_images).view(-1)    #Nx1->N
            real_loss = loss_fn(discr_pred_real , real_labels)
            
            
            # discriminator loss on fake images 
            discr_pred_fake = model.discriminator_forward(fake_images.detach()).view(-1)
            fake_loss = loss_fn(discr_pred_fake  , fake_labels )           #binary loss
            
            
            
            # combined loss
            discr_loss = 0.5*(real_loss + fake_loss)
            discr_loss.backward()
            optimizer_discr.step()
            
            #----------------------------------
            
            # Train generator
            
            #---------------------------------
            
            
            optimizer_gen.zero_grad()
            
            # get discriminator loss on fake images with flipped labels
            
            dis_pred_fake =  model.discriminator_forward(fake_images).view(-1)
            genner_loss = loss_fn(dis_pred_fake , flipped_fake_image)
            genner_loss.backward()
            
            optimizer_gen.step()
            
            #--------------------
            
            # logging 
            
            #----------------------
            
            log_dict['train_generator_loss_per_batch'].append(genner_loss.item())
            log_dict['train_discriminator_loss_per_batch'].append(discr_loss.item())
            
            predicted_labels_real =  torch.where(discr_pred_real.detach()>0.,1.,0.0)
            predicted_labels_fake = torch.where(discr_pred_fake.detach()>0.0,1.0,0.0)
            
            acc_real = (predicted_labels_real == real_labels).float().mean()*100.
            acc_fake = (predicted_labels_fake == fake_labels).float().mean()*100.
            
            
            
            log_dict['train_discriminator_real_acc_per_batch'].append(acc_real.item())
            log_dict['train_discriminator_fake_acc_per_batch'].append(acc_fake.item())  
            
            
            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Gen/Dis Loss: %.4f/%.4f' % (epoch+1, num_epochs, batch_idx, len(train_loader), genner_loss.item(), discr_loss.item()))
                print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
                
            # save images for evaluation 
            
            with torch.no_grad():
                fake_images = model.generator_forward(fixed_noise).detach().cpu()
                log_dict['images_from_noise_per_epoch'].append(
                torchvision.utils.make_grid(fake_images, padding=2, normalize=True))
            
                
                
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
        
    return log_dict