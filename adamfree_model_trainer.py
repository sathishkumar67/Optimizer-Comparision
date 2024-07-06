from modules import *
import numpy as np

@dataclass
class Args:
    epochs: int = 5
    lr: float = 3e-3
    batch_size: int = 64
    weight_decay: float = 1e-5
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_class: int = 128
    device_count: int = 1

# instantize the args
args = Args()

train_loss = []
val_loss = []

class AdamScheduleFree_Model(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        self.model.train()
        optimizer = self.optimizers()
        optimizer.train()
        optimizer.zero_grad()
        
        batch, label = batch
        out = self.model(batch)
        loss = F.cross_entropy(out, label)
        train_loss.append(loss.item())
        self.log("Train_Loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = AdamWScheduleFree(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        return optimizer
    
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        optimizer = self.optimizers()
        optimizer.eval()
        
        batch, label = batch
        out = self.model(batch)
        loss = F.cross_entropy(out, label)
        val_loss.append(loss.item())
        self.log("Val_Loss", loss, prog_bar=True)

        return loss
    
def main():
    # seed
    torch.manual_seed(args.seed)

    print("Training on", args.device)
    print("Preparing dataset...")
    # prepare dataset
    train_dataset = torchvision.datasets.Food101(root='./data', split='train', download=True, transform=transform)
    test_dataset = torchvision.datasets.Food101(root='./data', split='test', download=True, transform=transform)

    print("Preparing dataloader...")
    # prepare dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("Preparing model...")
    # prepare the model
    model = resnet152(pretrained=False)
    model.fc = nn.Linear(2048, args.num_class)

    adamschedulefree_model = AdamScheduleFree_Model(model=model)

    # training
    trainer = Trainer(max_epochs=args.epochs,
                  accelerator="cuda")
    trainer.fit(adamschedulefree_model, train_dataloader, test_dataloader)

    # save the model
    torch.save(adamschedulefree_model.model.state_dict(), "adamschedulefree_model.pt")

    # save the loss
    np.save("train_loss.npy", np.array(train_loss))
    np.save("val_loss.npy", np.array(val_loss))

    print("Done!")

if __name__ == "__main__":
    main()
