from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import os

# load dataset
digits = load_digits()

# create folder
os.makedirs("full_digit_dataset", exist_ok=True)

# save ALL images
for i in range(len(digits.images)):

    plt.imshow(digits.images[i], cmap='gray')

    plt.axis('off')

    plt.savefig(
        f"full_digit_dataset/image_{i}_label_{digits.target[i]}.png"
    )

    plt.close()

print("Full dataset saved successfully!")