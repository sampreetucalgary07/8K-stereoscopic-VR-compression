import torch
import torch.optim as optim
import os

##User defined libraries
from model.discriminator import Discriminator


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least loss, then save the
    model state.
    """

    def __init__(
        self,
        model_path,
        model_name,
        best_valid_loss=float("inf"),
        remove_previous_model=False,
    ):
        self.counter = 0
        if self.counter == 0:
            self.best_valid_loss = best_valid_loss

        self.model_path = model_path
        self.model_name = model_name
        if os.path.exists(self.model_path) and not remove_previous_model:
            raise ValueError(
                "Model path already exists! Set remove_previous_model=True to remove the previous model"
            )

    def save(
        self,
        current_valid_loss,
        epoch,
        model,
        optimizer_G,
        optimizer_D,
        discriminator,
        criterion,
        save_pth=True,
    ):
        self.counter += 1
        if save_pth:
            if optimizer_G is None or optimizer_D is None or criterion is None:
                raise ValueError(
                    "optimizers and criterion cannot be None when pth is True"
                )

        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\ncurrent_valid_loss: {current_valid_loss}")
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model of epoch: {epoch}\n")
            self.early_stopping_counter = 0

            if save_pth:
                self.model_save_path = os.path.join(
                    self.model_path, self.model_name + f".pth"
                )
                if os.path.exists(self.model_save_path):
                    os.remove(self.model_save_path)
                    print("Previously saved model deleted at the path")

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "discriminator_state_dict": discriminator.state_dict(),
                        "optimizer_G_state_dict": optimizer_G.state_dict(),
                        "optimizer_D_state_dict": optimizer_D.state_dict(),
                        "losses_used": criterion,
                        "best_loss_value": current_valid_loss,
                    },
                    self.model_save_path,
                )
                print("Model saved at: ", self.model_save_path)
            else:
                self.model_save_path = os.path.join(
                    self.model_path, self.model_name + f".pt"
                )
                if os.path.exists(self.model_save_path):
                    os.remove(self.model_save_path)
                    print("Previously saved model deleted at the path")
                torch.save(model.state_dict(), self.model_save_path)
                print("Model saved at: ", self.model_save_path)


# Example usage:
# Initialize the class with the path where you want to save the model
# save_best_model = SaveBestModel(model_path='./model/checkpoint.pth')

# During training, call the class with the necessary parameters
# save_best_model(current_valid_loss, epoch, generator, discriminator, optimizer_G, optimizer_D, criterion)


def loadModel(model_path, model_class, discriminator_class, pth=False, device="cuda"):
    model = model_class.to(device)
    discriminator = discriminator_class.to(device)
    if pth:
        checkpoint = torch.load(model_path, map_location=device)
        print(checkpoint.keys())
        try:
            discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        except:
            pass
        optimizer_G = optim.Adam(model.parameters(), lr=0.000001, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(
            discriminator.parameters(), lr=0.000001, betas=(0.5, 0.999)
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
        epoch = checkpoint["epoch"]
        try:
            losses_used = checkpoint["losses_used"]
        except:
            losses_used = None

        try:
            best_loss_value = checkpoint["best_loss_value"]
        except:
            best_loss_value = None

        print("\n ----- Loaded Model INFO ----- \n")
        print(f"Model loaded successfully from {model_path}")
        print(f"Model trained for {epoch} epochs")
        print(f"losses Used in model: {losses_used}")
        print(f"Model best loss value: {best_loss_value}")
        print(f"NOTE: model.eval() is pending! Please do it manually.")
        return (
            model,
            optimizer_G,
            optimizer_D,
            discriminator,
            epoch,
            losses_used,
            best_loss_value,
        )

    else:
        print(f"NOTE: model.eval() is pending! Please do it manually.")
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model, None, None, None, None


class EarlyStopping:
    def __init__(self, tolerance=3, min_loss=0.0):

        self.tolerance = tolerance
        self.min_loss = min_loss
        self.counter = 0
        self.early_stop = False

    def save(self, train_loss):
        if train_loss > self.min_loss:
            self.min_loss = train_loss
            self.counter += 1
            if self.counter >= self.tolerance:
                return True
        else:
            self.counter = 0

    # def __call__(self, train_loss, validation_loss):
    #     if (validation_loss - train_loss) > self.min_delta:
    #         self.counter += 1
    #         if self.counter >= self.tolerance:
    #             self.early_stop = True
