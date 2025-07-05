import os
import tqdm
import torch
import csv

def resume_models(generator, discriminator, output_dir, pretrain_epochs):
    models = os.listdir(output_dir)
    resume_pretrain_epoch = 0 
    resume_epoch = 0

    def get_latest(name):
        filtered = [m for m in models if m.startswith(name)]
        return max(filtered, key=lambda x: int(x.split('_')[-1].split('.')[0])) if filtered else None
    
    pretrain_generator = get_latest("pretrain")
    if pretrain_generator:
        resume_pretrain_epoch = int(pretrain_generator.split('_')[-1].split('.')[0])
        generator.load_state_dict(torch.load(os.path.join(output_dir, pretrain_generator)))
        tqdm.write(f"Resuming Pretraining from epoch {resume_pretrain_epoch}")

    generator_model = get_latest("generator")
    if generator_model:
        generator.load_state_dict(torch.load(os.path.join(output_dir, generator_model)))
        resume_epoch = int(generator_model.split('_')[-1].split('.')[0])
        resume_pretrain_epoch = pretrain_epochs
        tqdm.write(f"Resuming Generator from epoch {resume_epoch}")

    discriminator_model = get_latest("discriminator")
    if discriminator_model:
        discriminator.load_state_dict(torch.load(os.path.join(output_dir, discriminator_model)))
        resume_pretrain_epoch = pretrain_epochs
        tqdm.write(f"Resuming Discriminator from epoch {resume_epoch}")

    return resume_pretrain_epoch, resume_epoch

# Erase lines from epochs that are about to be rewritten
def resume_log(csv_file, resume_epoch):
    # Read all existing rows
    with open(csv_file, mode='r') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if int(row['epoch']) < resume_epoch]
        fieldnames = reader.fieldnames

    # Rewrite the file with filtered rows
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)