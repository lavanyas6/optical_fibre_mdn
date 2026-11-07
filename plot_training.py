
import pandas as pd
import matplotlib.pyplot as plt

# Load training history
history = pd.read_csv("mdn_run/training_history.csv")

# Plot total training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history["epoch"], history["train_loss"], label="Training loss")
plt.plot(history["epoch"], history["val_loss"], label="Validation loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MDN Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mdn_run/training_loss.png", dpi=300)
plt.show()

# Plot validation NLL
plt.figure(figsize=(10, 6))
plt.plot(history["epoch"], history["val_nll"], label="Validation NLL")

plt.xlabel("Epoch")
plt.ylabel("Negative log likelihood")
plt.title("MDN Validation NLL")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mdn_run/validation_nll.png", dpi=300)
plt.show()

# Plot pulse classification accuracy
plt.figure(figsize=(10, 6))
plt.plot(
    history["epoch"],
    history["val_accuracy"] * 100,
    label="Validation accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Pulse Classification Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mdn_run/pulse_accuracy.png", dpi=300)
plt.show()

print("Training plots saved in mdn_run/")
