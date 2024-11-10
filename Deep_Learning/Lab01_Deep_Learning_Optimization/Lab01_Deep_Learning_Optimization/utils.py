import numpy as np
import matplotlib.pyplot as plt

def plot_dense_network(size_inputs, n_hidden_layers, sizes_hidden_layers, size_outputs):
    # Make neurons coordinates
    x = 0
    input_layer = np.stack((x*np.ones(size_inputs), np.linspace(3/size_outputs, 10-3/size_inputs, num=size_inputs))).transpose()
    print_network = input_layer
    for n_hidden_layer in range(n_hidden_layers):
        x += 2
        n_neurons = sizes_hidden_layers[n_hidden_layer]
        layer = np.stack((x*np.ones(n_neurons), np.linspace(3/n_neurons, 10-3/n_neurons, num=n_neurons))).transpose()
        print_network = np.concatenate((print_network, layer))
    x += 2
    output_layer = np.stack((x*np.ones(size_outputs), np.linspace(3/size_outputs, 10-3/size_outputs, num=size_outputs))).transpose()
    print_network = np.concatenate((print_network, output_layer))
    
    # Plot neurons
    fig, ax = plt.subplots()
    ax.scatter(print_network[:,0], print_network[:,1], c=print_network[:,0], s=5e2, edgecolors='none', zorder=2)
    
    legends = ['Input layer'] + ['Hidden layer {}'.format(i+1) for i in range(n_hidden_layers)] + ['Output layer']
    x = 0
    for it, legend in enumerate(legends):
        plt.text(x-1, -1-it%2, legend, size=10)
        x += 2
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-2,x])
    ax.set_ylim([-3,11])
    
    # Connect neurons 
    x = 0
    for n_hidden_layer in range(n_hidden_layers+2):
        for neuron in print_network[np.where(print_network[:,0]==x)]:
            for neuron2 in print_network[np.where(print_network[:,0]==x+2)]:
                points = np.stack((neuron, neuron2)).transpose()
                ax.plot(points[0], points[1], 'k-', zorder=1)
        x += 2
    ax.set_title('Dense neural network architecture', fontsize=16)
    plt.tight_layout
    plt.show()
