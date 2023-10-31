import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
import os

# Load the pre-trained ResNet-18 model
model = models.resnet18(weights=models.resnet.ResNet18_Weights.DEFAULT)

# Set the model to evaluation mode (no training)
model.eval()

# Define the image transformation
preprocess = transforms.Compose([
    transforms.Resize(400),  # Increase the image size
    transforms.CenterCrop(400),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to perform image recognition
def recognize_image():
    file_path = os.path.abspath(filedialog.askopenfilename())
    if file_path:
        image = Image.open(file_path)
        image = preprocess(image)
        image = image.unsqueeze(0)  # Add a batch dimension

        # Make a prediction
        with torch.no_grad():
            output = model(image)

        # Get the class probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get the top 5 predicted classes and their probabilities
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        class_labels = [class_names[top5_catid[i]] for i in range(top5_prob.size(0))]

        # Display the recognition results
        results_label.config(text="\nTop 5 clases mas reconocidas:")
        for i in range(top5_prob.size(0)):
            results_label.config(text=results_label.cget("text") + f"\n{class_labels[i]}: {top5_prob[i].item() * 100:.2f}%")

        # Display the image
        img = Image.open(file_path)
        img.thumbnail((800, 600))  # Resize the image to fit the label
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img

# Create the main GUI window
root = tk.Tk()
root.title("ResNet Reconocimiento de imagenes")
root.geometry("800x600")  # Set window size
root.option_add('*Font', 'Helvetica 14 bold')  # Set a bold font

# Button to trigger image recognition
recognize_button = Button(root, text="RECONOCER IMAGEN", command=recognize_image)
recognize_button.pack()

# Label for recognition results
results_label = Label(root, text="")
results_label.pack()

# Create a label for displaying the image
img_label = Label(root)
img_label.pack()

# Load the ImageNet class labels
with open("imagenet-classes.txt") as f:
    class_names = [line.strip() for line in f.readlines()]

# Start the GUI main loop
root.mainloop()
