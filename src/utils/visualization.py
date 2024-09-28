import matplotlib.pyplot as plt

def plot_sample_image(features, target, index):
    plt.imshow(features[index].reshape(28, 28), cmap='gray')
    plt.axis("off")
    plt.title(str(target[index]))
    plt.savefig('graph.png')
    plt.show()



def plot_loss_accuracy(iteration_list, loss_list, accuracy_list):
    # Plot loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(iteration_list, loss_list)
    plt.xlabel("İterasyon Sayisi")
    plt.ylabel("Kayip")
    plt.title("CNN: Kayip Grafiği")

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(iteration_list, accuracy_list, color="red")
    plt.xlabel("İterasyon Sayisi")
    plt.ylabel("Dogruluk")
    plt.title("CNN: Dogruluk Grafiği")

    plt.tight_layout()
    plt.show()
