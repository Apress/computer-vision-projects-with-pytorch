## imports from existing config
from config import *
from torch.utils.data import DataLoader
## importing base dataset from the dataset class
from dataset import BaseDataset


def train_generator(train_dataloader, epochs) -> None:
    ## starting with train generator
    ## defining data loader
    ## defining the loss function
    batch_count = len(train_dataloader)
    ## start training for generator block
    generator.train()

    for index, (lr, hr) in enumerate(train_dataloader):
        ## getting hr to cuda or cpu
        hr = hr.to(device)
        ## getting lr to cuda or cpu
        lr = lr.to(device)
        
        ## initializing to zero grad for generator to avoid gradient accumulation
        ## Accumulation is only suggested in case of time base model
        generator.zero_grad()

        sr = generator(lr)
        ## defining pixel loss
        pixel_losses = pixel_criterion(sr, hr)
        ## get step function from optimisers
        pixel_losses.backward()
        ## adam optimisers for generator
        p_optimizer.step()

        iteration = index + epochs * batch_count + 1
        writer.add_scalar("Train_Generator/Loss", pixel_losses.item(), iteration)

        if (index + 1) % 10 == 0 or (index + 1) == batch_count:
            print(f"Train Epoch[{epochs + 1:04d}/{p_epochs:04d}]({index + 1:05d}/{batch_count:05d}) "
                  f"Loss: {pixel_losses.item():.6f}.")


def train_adversarial(train_dataloader, epoch) -> None:
    ## for training adversarial network

    batches = len(train_dataloader)
    ## training discriminator and generator
    discriminator.train()
    generator.train()

    for index, (lr, hr) in enumerate(train_dataloader):
        hr = hr.to(device)
        lr = lr.to(device)
        
        label_size = lr.size(0)
        fake_label = torch.full([label_size, 1], 0.0, dtype=lr.dtype, device=device)
        real_label = torch.full([label_size, 1], 1.0, dtype=lr.dtype, device=device)
        

        ## initializing zero grad since we want to avoid grad accumulation
        discriminator.zero_grad()

        output_dis = discriminator(hr)
        dis_loss_hr = adversarial_criterion(output_dis, real_label)
        dis_loss_hr.backward()
        dis_hr = output_dis.mean().item()

        sr = generator(lr)

        output_dis = discriminator(sr.detach())
        dis_loss_sr = adversarial_criterion(output_dis, fake_label)
        dis_loss_sr.backward()
        dis_sr1 = output_dis.mean().item()

        dis_loss = dis_loss_hr + dis_loss_sr
        d_optimizer.step()


        generator.zero_grad()

        output = discriminator(sr)

        pixel_loss = pixel_weight * pixel_criterion(sr, hr.detach())
        perceptual_loss = content_weight * content_criterion(sr, hr.detach())
        adversarial_loss = adversarial_weight * adversarial_criterion(output, real_label)

        gen_loss = pixel_loss + perceptual_loss + adversarial_loss
        gen_loss.backward()
        g_optimizer.step()
        dis_sr2 = output.mean().item()


        iteration = index + epoch * batches + 1
        writer.add_scalar("Train_Adversarial/D_Loss", dis_loss.item(), iteration)
        writer.add_scalar("Train_Adversarial/G_Loss", gen_loss.item(), iteration)
        writer.add_scalar("Train_Adversarial/D_HR", dis_hr, iteration)
        writer.add_scalar("Train_Adversarial/D_SR1", dis_sr1, iteration)
        writer.add_scalar("Train_Adversarial/D_SR2", dis_sr2, iteration)

        if (index + 1) % 10 == 0 or (index + 1) == batches:
            print(f"Training stage - adversarial loss"
                  f"Epoch[{epoch + 1:04d}/{epochs:04d}]({index + 1:05d}/{batches:05d}) "
                  f"D Loss: {dis_loss.item():.6f} G Loss: {gen_loss.item():.6f} "
                  f"D(HR): {dis_hr:.6f} D(SR1)/D(SR2): {dis_sr1:.6f}/{dis_sr2:.6f}.")


def validate(valid_dataloader, epoch, stage) -> float:
    ## defining validation function
    ## defining loss 

    batch_count = len(valid_dataloader)

    generator.eval()

    total_psnr_value_sum = 0.0

    with torch.no_grad():
        for index, (lr, hr) in enumerate(valid_dataloader):
            hr = hr.to(device)
            lr = lr.to(device)
            sr = generator(lr)

            

            mse_losses = psnr_criterion(sr, hr)
            psnr_val = 10 * torch.log10(1 / mse_losses).item()
            total_psnr_value_sum += psnr_val

        aveg_psnr_value = total_psnr_value_sum / batch_count

        if stage == "generator":
            writer.add_scalar("Val_Generator/PSNR", aveg_psnr_value, epoch + 1)
        elif stage == "adversarial":
            writer.add_scalar("Val_Adversarial/PSNR", aveg_psnr_value, epoch + 1)

        print(f"Valid stage: {stage} Epoch[{epoch + 1:04d}] avg PSNR: {aveg_psnr_value:.2f}.\n")

    return aveg_psnr_value


def main() -> None:

    ## creating directories 
    ## making up training andf validation datasets loc
    ## check for training 
    ## check for resuming training if an opportunity
    if not os.path.exists(exp_dir1):
        os.makedirs(exp_dir1)
    if not os.path.exists(exp_dir2):
        os.makedirs(exp_dir2)


    train_dataset = BaseDataset(train_dir, image_size, upscale_factor, "train")
    train_dataloader = DataLoader(train_dataset, batch_size, True, pin_memory=True)

    valid_dataset = BaseDataset(valid_dir, image_size, upscale_factor, "valid")
    valid_dataloader = DataLoader(valid_dataset, batch_size, False, pin_memory=True)
    
    

    if resume:
       ## for resuming training 
        if resume_p_weight != "":
            generator.load_state_dict(torch.load(resume_p_weight))
        else:
            discriminator.load_state_dict(torch.load(resume_d_weight))
            generator.load_state_dict(torch.load(resume_g_weight))


    best_psnr_val = 0.0

    for epoch in range(start_p_epoch, p_epochs):

        train_generator(train_dataloader, epoch)

        psnr_val = validate(valid_dataloader, epoch, "generator")

        best_condition = psnr_val > best_psnr_val
        best_psnr_val = max(psnr_val, best_psnr_val)

        torch.save(generator.state_dict(), os.path.join(exp_dir1, f"p_epoch{epoch + 1}.pth"))
        if best_condition:
            torch.save(generator.state_dict(), os.path.join(exp_dir2, "p-best.pth"))

    ## saving best model
    torch.save(generator.state_dict(), os.path.join(exp_dir2, "p-last.pth"))


    best_psnr_val = 0.0

    generator.load_state_dict(torch.load(os.path.join(exp_dir2, "p-best.pth")))

    for epoch in range(start_epoch, epochs):

        train_adversarial(train_dataloader, epoch)

        psnr_val = validate(valid_dataloader, epoch, "adversarial")

        best_condition = psnr_val > best_psnr_val
        best_psnr_val = max(psnr_val, best_psnr_val)

        torch.save(discriminator.state_dict(), os.path.join(exp_dir1, f"d_epoch{epoch + 1}.pth"))
        torch.save(generator.state_dict(), os.path.join(exp_dir1, f"g_epoch{epoch + 1}.pth"))
        if best_condition:
            torch.save(discriminator.state_dict(), os.path.join(exp_dir2, "d-best.pth"))
            torch.save(generator.state_dict(), os.path.join(exp_dir2, "g-best.pth"))

        d_scheduler.step()
        g_scheduler.step()


    torch.save(discriminator.state_dict(), os.path.join(exp_dir2, "d-last.pth"))
    torch.save(generator.state_dict(), os.path.join(exp_dir2, "g-last.pth"))


if __name__ == "__main__":
    main()
