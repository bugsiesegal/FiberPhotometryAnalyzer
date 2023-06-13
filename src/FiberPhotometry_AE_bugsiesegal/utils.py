import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output, display
from ipywidgets import IntSlider, Output, VBox, Layout, interactive


class InteractiveWindowExplorer:
    def __init__(self, dataset, delay=0.5):
        """
        Constructor for the InteractiveWindowExplorer class.

        Parameters:
        dataset: The dataset which windows are to be explored.
        delay: Delay between plot updates to prevent crashing. Default is 0.1 seconds.
        """
        self.dataset = dataset
        self.delay = delay
        self._lock = threading.Lock()
        self.out = Output()

    def plot_window(self, idx):
        """
        Plots a window from the dataset at the specified index.

        Parameters:
        idx: The index of the window.
        """
        with self._lock:
            time.sleep(self.delay)

            with self.out:
                clear_output(wait=True)  # Clear previous plot

                window = self.dataset[idx]
                sampling_times = self.dataset.get_sample_times(idx)

                # Plotting
                plt.figure(figsize=(10, 6))
                plt.plot(sampling_times, window, label=f"Window {idx}")
                plt.title(f"Time Series Window {idx}")
                plt.xlabel('Sampling Times (in seconds)')
                plt.ylabel('Time Series Data')
                plt.legend()

                plt.show()

    def interactive_plot(self):
        """
        Creates an interactive plot of the dataset windows.
        """
        slider = IntSlider(min=0, max=len(self.dataset) - 1, step=1, continuous_update=False,
                           layout=Layout(width='auto'))

        interactive_plot = interactive(self.plot_window, idx=slider)

        # Now place the interactive plot widget in the VBox list
        display(VBox([self.out, interactive_plot]))


def visualize_neurons(model, encoding_size, input_dim, num_iterations=5000, lr=0.01):
    # Determine the number of rows and columns for the grid
    grid_rows = int(np.sqrt(encoding_size))
    grid_cols = int(np.ceil(encoding_size / grid_rows))

    # Create a figure
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(10, 10))

    # Perform feature visualization for each neuron
    for neuron_index in range(encoding_size):
        # Create a tensor with random values of the size of the input dimension
        random_input = torch.randn(input_dim).requires_grad_()

        # Define the optimizer
        optimizer = torch.optim.Adam([random_input], lr=lr)

        # Perform gradient ascent
        for i in range(num_iterations):
            optimizer.zero_grad()
            output = model.encoder(random_input.unsqueeze(0))
            loss = -output[0][neuron_index]
            loss.backward()
            optimizer.step()

        # Detach the input from the computational graph and convert it to a numpy array
        optimized_input = random_input.detach().numpy()

        # Plot the optimized input
        ax = axes[neuron_index // grid_cols, neuron_index % grid_cols]
        ax.imshow(optimized_input)
        ax.set_title(f'Neuron {neuron_index + 1}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
