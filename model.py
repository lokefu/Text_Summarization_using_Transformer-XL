import torch
import torch.utils.data as data
from tokenization_transfo_xl import TransfoXLTokenizer
from modeling_transfo_xl import TransfoXLModel
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader


# Load the pre-trained Transformer-XL model and tokenizer
print("Loading Transformer-XL model and tokenizer...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
model = TransfoXLModel.from_pretrained('transfo-xl-wt103', output_hidden_states=True, output_attentions=True)
model.to(device)


# Load the CNN/Daily Mail news dataset
print("Loading CNN/Daily Mail news dataset...")
train_dataset = torch.load('train_subset.pt')
val_dataset = torch.load('val_subset.pt')


# Set batch size and collate function 
batch_size = 1
def collate_fn(batch):
   
    input_ids = torch.tensor([example['input_ids'] for example in batch])
    attention_mask = torch.tensor([example['attention_mask'] for example in batch])
    labels = torch.tensor([example['labels'] for example in batch])
    return {'input_ids': input_ids.to(device), 'attention_mask': attention_mask.to(device), 'labels': labels.to(device)}


# Create data loaders
print("Creating data loaders...")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=0,collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)


# Set optimization parameters
print("Setting optimization parameters...")
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1, num_training_steps=len(train_dataloader)*1) #100,10


# Add model checkpoint saving
best_val_loss = float('inf')
for epoch in range(2): #3
    print(f"Training epoch {epoch}...")
    model.train()
    total_loss = 0
    num_batches = 0
    for batch in train_dataloader:

        input_ids = torch.tensor(batch['input_ids']).clone().detach().to(device)
        attention_mask = torch.tensor(batch['attention_mask']).clone().detach().to(device)
        labels = torch.tensor(batch['labels']).clone().detach().to(device)

        outputs = model(input_ids=input_ids)
        last_hidden_state = outputs.last_hidden_state

        # Generate the summary by taking the first 3 sentences of the input
        summary_input = last_hidden_state[:, :3, :]
        summary_output = last_hidden_state[:, :128, :].mean(dim=1) # Average pooling over the 128 tokens
        loss = torch.mean(torch.square(summary_output - summary_input))

        loss.backward()


        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1

    avg_train_loss = total_loss / num_batches

    # Evaluate the model on the validation set
    print("Evaluating on validation set...")
    model.eval()
    total_val_loss = 0
    num_val_batches = 0
    with torch.no_grad():
        for batch in val_dataloader:

            input_ids = torch.tensor(batch['input_ids']).clone().detach().to(device)
            attention_mask = torch.tensor(batch['attention_mask']).clone().detach().to(device)
            labels = torch.tensor(batch['labels']).clone().detach().to(device)

            # Convert the batch_input_ids to a tensor and reshape it
            input_ids = torch.tensor(input_ids).view(batch_size, -1)

            outputs = model(input_ids=input_ids)
            last_hidden_state = outputs.last_hidden_state

            # Generate the summary by taking the first 3 sentences of the input
            summary_input = last_hidden_state[:, :3, :]
            summary_output = last_hidden_state[:, :128, :].mean(dim=1) # Average pooling over the 128 tokens
            loss = torch.mean(torch.square(summary_output - summary_input))

            total_val_loss += loss.item()
            num_val_batches += 1

        avg_val_loss = total_val_loss / num_val_batches

        # Add early stopping and model checkpoint saving
        if avg_val_loss < best_val_loss:
            print("Saving best checkpoint...")
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, 'best_checkpoint.pt')
            tokenizer.save_pretrained('best_checkpoint')
        else:
            break

    print(f'Epoch {epoch}: train loss = {avg_train_loss}, validation loss = {avg_val_loss}')

