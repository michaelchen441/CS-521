import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json
import os 
import matplotlib.pylab as plt


def get_predictions():
    # Load the pre-trained ResNet18 model
    model = models.resnet18(pretrained=True)
    model.eval()  # Set model to evaluation mode

    # Define the image preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]   
        )
    ])

    # Load the ImageNet class index mapping
    with open("imagenet_class_index.json") as f:
        class_idx = json.load(f)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    idx2synset = [class_idx[str(k)][0] for k in range(len(class_idx))]
    id2label = {v[0]: v[1] for v in class_idx.values()}

    imagenet_path = './imagenet_samples'

    # List of image file paths
    image_paths = os.listdir(imagenet_path)
    print("image_paths:", image_paths)
    images = []
    labels = []

    for img_path in image_paths:
        # Open and preprocess the image
        # my_img = os.path.join(img_path, os.listdir(img_path)[2])
        my_img = os.path.join(imagenet_path, img_path)
        input_image = Image.open(my_img).convert('RGB')
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

        # Move the input and model to GPU if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        # Perform inference
        with torch.no_grad():
            output = model(input_batch)

        # Get the predicted class index
        _, predicted_idx = torch.max(output, 1)
        images.append(input_batch)
        labels.append(predicted_idx)

        predicted_idx = predicted_idx.item()
        predicted_synset = idx2synset[predicted_idx]
        predicted_label = idx2label[predicted_idx]


        print(f"Predicted label: {predicted_synset} ({predicted_label})")

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)

    return model, images, labels

def smoothgrad(model, images, labels, noise, sample_size):
    model.eval()
    images = images.requires_grad_()

    sens_map = torch.zeros_like(images)

    #average sample_size num of inputs of randomly smoothed versions of the map
    for i in range(sample_size):
        #add gaussian smooth to x with stddev= noise input
        smoothed = images + (torch.randn_like(images) * noise)
        out_smooth = model(smoothed)

        out_scores = out_smooth.gather(1, labels.view(-1,1)).squeeze()

        model.zero_grad()
        out_scores.backward(torch.ones_like(out_scores))

        #add each to the sensitivity map and reset gradients
        sens_map += images.grad.data
        images.grad.zero_()

    sens_map /= sample_size
    sens_map = torch.abs(sens_map) #paper suggests abs for imagenet
    # sens_map *= images 
    #TODO clamp by 99th percentile - paper suggestion
    return sens_map

def main():
    imagenet_path = './imagenet_samples'
    image_paths = os.listdir(imagenet_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])
    
    model, images, labels = get_predictions()

    sens_maps = smoothgrad(model, images, labels, .5, 50)

    fig, axes = plt.subplots(len(images), 2)

    for i in range(len(images)):
        maps = sens_maps[i].cpu().numpy().transpose(1, 2, 0) 
        maps = (maps - maps.min()) / (maps.max() - maps.min())  # Normalize

        axes[i,0].imshow(maps)
        axes[i,0].axis('off')
        # plt.title(f"SmoothGrad for Image {i+1}")

        my_img = os.path.join(imagenet_path, image_paths[i])
        input_image = Image.open(my_img).convert('RGB')
        input_image = preprocess(input_image)
        # plt.subplot(1,2,i+1)
        # plt.imshow(input_image)
        axes[i,1].imshow(input_image)
        axes[i,1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()
