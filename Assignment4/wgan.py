from helper_evaluate import compute_accuracy
from helper_evaluate import compute_epoch_loss_classifier
from helper_evaluate import compute_epoch_loss_autoencoder

import time
import torch
import torch.nn.functional as F
import torchvision
import torch.autograd




def train_wgan_v1(num_epochs, model, optimizer_gen, optimizer_discr, 
                  latent_dim, device, train_loader,
                  discr_iter_per_generator_iter=5,
                  logging_interval=100, 
                  gradient_penalty=False,
                  gradient_penalty_weight=10,
                  save_model=None):
    
    log_dict = {'train_generator_loss_per_batch': [],
                'train_discriminator_loss_per_batch': [],
                'train_discriminator_real_acc_per_batch': [],
                'train_discriminator_fake_acc_per_batch': [],
                'images_from_noise_per_epoch': []}

    if gradient_penalty:
        log_dict['train_gradient_penalty_loss_per_batch'] = []

    def loss_fn(y_pred, y_true):
        return -torch.mean(y_pred * y_true)

    # Batch of latent (noise) vectors for
    # evaluating / visualizing the training progress
    # of the generator
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device) # format NCHW

    start_time = time.time()
    
    
    skip_generator = 1
    for epoch in range(num_epochs):

        model.train()
        
        for batch_idx, (features, _) in enumerate(train_loader):

            batch_size = features.size(0)

            # real images
            real_images = features.to(device)
            real_labels = torch.ones(batch_size, device=device) # real label = 1

            # generated (fake) images
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)  # format NCHW
            fake_images = model.generator_forward(noise)
            
            fake_labels = -real_labels # fake label = -1    
            flipped_fake_labels = real_labels # here, fake label = 1

    
            # --------------------------
            # Train Discriminator
            # --------------------------

            optimizer_discr.zero_grad()

            # get discriminator loss on real images
            discr_pred_real = model.discriminator_forward(real_images).view(-1) # Nx1 -> N
            real_loss = loss_fn(discr_pred_real, real_labels)
            # real_loss.backward()

            # get discriminator loss on fake images
            discr_pred_fake = model.discriminator_forward(fake_images.detach()).view(-1)
            fake_loss = loss_fn(discr_pred_fake, fake_labels)
            # fake_loss.backward()
            
            # combined loss
            discr_loss = 0.5*(real_loss + fake_loss)

            ###################################################
            # gradient penalty
            if gradient_penalty:

                alpha = torch.rand(batch_size, 1, 1, 1).to(device)

                interpolated = alpha * real_images + (1 - alpha) * fake_images.detach()
                interpolated.requires_grad = True

                discr_out = model.discriminator_forward(interpolated)

                grad_values = torch.ones(discr_out.size()).to(device)
                gradients = torch.autograd.grad(
                    outputs=discr_out,
                    inputs=interpolated,
                    grad_outputs=grad_values,
                    create_graph=True,
                    retain_graph=True)[0]

                gradients = gradients.view(batch_size, -1)

                # calc. norm of gradients, adding epsilon to prevent 0 values
                epsilon = 1e-13
                gradients_norm = torch.sqrt(
                    torch.sum(gradients ** 2, dim=1) + epsilon)

                gp_penalty_term = ((gradients_norm - 1) ** 2).mean() * gradient_penalty_weight
                discr_loss += gp_penalty_term
                
                log_dict['train_gradient_penalty_loss_per_batch'].append(gp_penalty_term.item())
            #######################################################
            
            discr_loss.backward()

            optimizer_discr.step()
            
            # Use weight clipping (standard Wasserstein GAN)
            if not gradient_penalty:
                for p in model.discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

            
            if skip_generator <= discr_iter_per_generator_iter:
                
                # --------------------------
                # Train Generator
                # --------------------------

                optimizer_gen.zero_grad()

                # get discriminator loss on fake images with flipped labels
                discr_pred_fake = model.discriminator_forward(fake_images).view(-1)
                gener_loss = loss_fn(discr_pred_fake, flipped_fake_labels)
                gener_loss.backward()

                optimizer_gen.step()
                
                skip_generator += 1
                
            else:
                skip_generator = 1
                gener_loss = torch.tensor(0.)

            # --------------------------
            # Logging
            # --------------------------   
            log_dict['train_generator_loss_per_batch'].append(gener_loss.item())
            log_dict['train_discriminator_loss_per_batch'].append(discr_loss.item())
            
            predicted_labels_real = torch.where(discr_pred_real.detach() > 0., 1., 0.)
            predicted_labels_fake = torch.where(discr_pred_fake.detach() > 0., 1., 0.)
            acc_real = (predicted_labels_real == real_labels).float().mean()*100.
            acc_fake = (predicted_labels_fake == fake_labels).float().mean()*100.
            log_dict['train_discriminator_real_acc_per_batch'].append(acc_real.item())
            log_dict['train_discriminator_fake_acc_per_batch'].append(acc_fake.item())         
            
            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Gen/Dis Loss: %.4f/%.4f' 
                       % (epoch+1, num_epochs, batch_idx, 
                          len(train_loader), gener_loss.item(), discr_loss.item()))

        ### Save images for evaluation
        with torch.no_grad():
            fake_images = model.generator_forward(fixed_noise).detach().cpu()
            log_dict['images_from_noise_per_epoch'].append(
                torchvision.utils.make_grid(fake_images, padding=2, normalize=True))


        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    
    return log_dict