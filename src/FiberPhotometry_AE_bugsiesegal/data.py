from typing import Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset


class FiberPhotometryDataset(Dataset):
    """
    A custom Dataset class for handling Fiber Photometry data.
    Fiber Photometry is a technique for recording neural activity.
    This class transforms time series data into windows for use in machine learning models.

    Attributes:
    time_series (Tensor): The raw time series data in windowed form.
    sampling_times (Tensor): The times at which the data was sampled, also in windowed form.
    transform (callable, optional): Optional transform to be applied on a sample.

    """

    def __init__(self, time_series: np.ndarray | torch.Tensor, sampling_times: np.ndarray | torch.Tensor,
                 window_size: int, step_size: int,
                 transform: Optional[Callable] = None) -> None:
        """
        Constructor for the FiberPhotometryDataset class.

        Parameters:
        time_series (Union[np.ndarray, torch.Tensor]): The raw time series data.
        sampling_times (Union[np.ndarray, torch.Tensor]): The times at which the data was sampled.
        window_size (int): The size of the windows into which the data is split.
        step_size (int): The step size for the window.
        transform (callable, optional): Optional transform to be applied on a sample.

        """
        super().__init__()

        self.original_time_series = time_series
        self.original_sampling_times = sampling_times

        if isinstance(time_series, np.ndarray):
            time_series = torch.from_numpy(time_series)
        if isinstance(sampling_times, np.ndarray):
            sampling_times = torch.from_numpy(sampling_times)

        self.time_series = time_series.unfold(0, window_size, step_size)
        self.sampling_times = sampling_times.unfold(0, window_size, step_size)
        self.transform = transform

    def normalize_windows(self) -> None:
        """
        Normalizes each window in the dataset independently to a range between 0 and 1.
        """
        self.time_series = (self.time_series - self.time_series.min(dim=1, keepdim=True)[0]) / \
                           (self.time_series.max(dim=1, keepdim=True)[0] - self.time_series.min(dim=1, keepdim=True)[
                               0] + 1e-8)

    def __len__(self) -> int:
        """
        Computes the length of the dataset.

        Returns:
        int: The number of windows in the dataset.

        """
        return self.time_series.shape[0]

    def get_sample_times(self, idx: int) -> torch.Tensor:
        """
        Retrieves the sampling times for a specific window of the time series data.

        Parameters:
        idx (int): The index of the window.

        Returns:
        Tensor: The corresponding window of the sampling times.

        """
        return self.sampling_times[idx]

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieves a window from the dataset at the specified index.

        Parameters:
        idx (int): The index of the window.

        Returns:
        Tensor: A window of the time series data.
        If transform is defined, the window data will be transformed.

        """
        window = self.time_series[idx]

        if self.transform:
            window = self.transform(window)

        return window


