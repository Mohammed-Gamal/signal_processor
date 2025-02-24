"""
-----------------------
Signal Processor Module
-----------------------
This module contains a bunch of methods for generating and processing discrete-time signals, including
    - Impulse sequence
    - Step sequence
    - Signal addition
    - Signal multiplication
    - Signal shifting
    - Signal folding
    - Even-odd decomposition
    - Plotting signals

Created on 02/10/2025 by Mohamed Gamal
Last updated on 02/24/2025 by Mohamed Gamal
"""

import numpy as np
import matplotlib.pyplot as plt

class SignalProcessor:
    """
    A class for generating and processing discrete-time signals, including impulse, step,
    addition, multiplication, shifting, and folding operations.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def impseq(k: int, n1: int, n2: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates a unit impulse sequence δ(n - k) over the range n1 <= n <= n2.

        Parameters:
        - k: int, the location of the impulse.
        - n1: int, the starting index of the sequence.
        - n2: int, the ending index of the sequence.

        Returns:
        - x: np.ndarray, the unit impulse sequence.
        - n: np.ndarray, the range of indices.

        Example:
        >>> x, n = SignalProcessor.impseq(0, -5, 5)
        >>> x
        array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        >>> n
        array([-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5])
        """
        n = np.arange(n1, n2 + 1)
        x = (n == k).astype(int)
        return x, n


    @staticmethod
    def stepseq(k: int, n1: int, n2: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates a unit step sequence u(n - k) over the range n1 <= n <= n2.

        Parameters:
        - k: int, the location of the step.
        - n1: int, the starting index of the sequence.
        - n2: int, the ending index of the sequence.

        Returns:
        - x: np.ndarray, the unit step sequence.
        - n: np.ndarray, the range of indices.

        Example:
        >>> x, n = SignalProcessor.stepseq(0, -5, 5)
        >>> x
        array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        >>> n
        array([-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5])
        """
        n = np.arange(n1, n2 + 1)
        x = (n >= k).astype(int)
        return x, n


    @staticmethod
    def sigadd(x1: np.ndarray, n1: np.ndarray, x2: np.ndarray, n2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Adds two discrete-time signals, y(n) = x1(n1) + x2(n2).

        Parameters:
        - x1: np.ndarray, the first signal.
        - n1: np.ndarray, the range of indices for the first signal.
        - x2: np.ndarray, the second signal.
        - n2: np.ndarray, the range of indices for the second signal.

        Returns:
        - y: np.ndarray, the sum of the two signals.
        - n: np.ndarray, the range of indices.

        Example:
        >>> x1 = np.array([1, 2, 3])
        >>> n1 = np.array([0, 1, 2])
        >>> x2 = np.array([4, 5, 6])
        >>> n2 = np.array([0, 1, 2])
        >>> y, n = SignalProcessor.sigadd(x1, n1, x2, n2)
        >>> y
        array([5, 7, 9])
        >>> n
        array([0, 1, 2])
        """
        # Align signals
        y1, y2, n = SignalProcessor.alignsigs(x1, n1, x2, n2)
        y = y1 + y2
        return y, n


    @staticmethod
    def sigmult(x1: np.ndarray, n1: np.ndarray, x2: np.ndarray, n2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Multiplies two discrete-time signals, y(n) = x1(n1) * x2(n2).

        Parameters:
        - x1: np.ndarray, the first signal.
        - n1: np.ndarray, the range of indices for the first signal.
        - x2: np.ndarray, the second signal.
        - n2: np.ndarray, the range of indices for the second signal.

        Returns:
        - y: np.ndarray, the product of the two signals.
        - n: np.ndarray, the range of indices.

        Example:
        >>> x1 = np.array([1, 2, 3])
        >>> n1 = np.array([0, 1, 2])
        >>> x2 = np.array([4, 5, 6])
        >>> n2 = np.array([0, 1, 2])
        >>> y, n = SignalProcessor.sigmult(x1, n1, x2, n2)
        >>> y
        array([ 4, 10, 18])
        >>> n
        array([0, 1, 2])
        """
        # Align signals
        y1, y2, n = SignalProcessor.alignsigs(x1, n1, x2, n2)
        y = y1 * y2
        return y, n


    @staticmethod
    def sigshift(x: np.ndarray, n: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Shifts a discrete-time signal by k units, y(n) = x(n - k).

        Parameters:
        - x: np.ndarray, the input signal.
        - n: np.ndarray, the range of indices for the input signal.
        - k: int, the number of units to shift the signal.
        
        Returns:
        - y: np.ndarray, the shifted signal.
        - n_shifted: np.ndarray, the range of indices for the shifted signal.

        Example:
        >>> x = np.array([1, 2, 3])
        >>> n = np.array([0, 1, 2])
        >>> y, n_shifted = SignalProcessor.sigshift(x, n, 2)
        >>> y
        array([0, 0, 1, 2, 3])
        >>> n_shifted
        array([0, 1, 2, 3, 4])
        """
        n_shifted = n + k
        y = np.array(x)
        return y, n_shifted

    @staticmethod
    def sigfold(x: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Folds (flips) a discrete-time signal, y(n) = x(-n).

        Parameters:
        - x: np.ndarray, the input signal.
        - n: np.ndarray, the range of indices for the input signal.

        Returns:
        - y: np.ndarray, the folded signal.
        - n_folded: np.ndarray, the range of indices for the folded signal.

        Example:
        >>> x = np.array([1, 2, 3])
        >>> n = np.array([0, 1, 2])
        >>> y, n_folded = SignalProcessor.sigfold(x, n)
        >>> y
        array([3, 2, 1])
        >>> n_folded
        array([-2, -1,  0])
        """
        y = np.flip(x)
        n_folded = -np.flip(n)
        return y, n_folded
    
    @staticmethod
    def alignsigs(x1: np.ndarray, n1: np.ndarray, x2: np.ndarray, n2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Aligns two signals by zero-padding to match their indices.

        Parameters:
        - x1: np.ndarray, the first signal.
        - n1: np.ndarray, the range of indices for the first signal.
        - x2: np.ndarray, the second signal.
        - n2: np.ndarray, the range of indices for the second signal.

        Returns:
        - x1_ext: np.ndarray, the first signal with zero-padding.
        - x2_ext: np.ndarray, the second signal with zero-padding.
        - n: np.ndarray, the range of indices.

        Example:
        >>> x1 = np.array([1, 2, 3])
        >>> n1 = np.array([0, 1, 2])
        >>> x2 = np.array([4, 5, 6])
        >>> n2 = np.array([1, 2, 3])
        >>> x1_ext, x2_ext, n = SignalProcessor.alignsigs(x1, n1, x2, n2)
        >>> x1_ext
        array([0, 1, 2, 3])
        >>> x2_ext
        array([0, 4, 5, 6])
        >>> n
        array([0, 1, 2, 3])
        """
        # Define common index range
        n = np.arange(min(min(n1), min(n2)), max(max(n1), max(n2)) + 1)

        x1_ext = np.zeros_like(n, dtype=float)
        x2_ext = x1_ext.copy()

        mask1 = (n >= n1[0]) & (n <= n1[-1])
        mask2 = (n >= n2[0]) & (n <= n2[-1])

        x1_ext[np.where(mask1)[0]] = x1[np.where(n1 == n[mask1])]
        x2_ext[np.where(mask2)[0]] = x2[np.where(n2 == n[mask2])]

        return x1_ext, x2_ext, n

    @staticmethod
    def even_odd_decomp(x: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decomposes a real-valued discrete-time signal into its even and odd components.

        Parameters:
        - x: np.ndarray, the input signal.
        - n: np.ndarray, the range of indices for the input signal.

        Returns:
        - x_e: np.ndarray, the even part of the signal.
        - x_o: np.ndarray, the odd part of the signal.
        - n_ext: np.ndarray, the range of indices.

        Example:
        >>> x = np.array([1, 2, 3])
        >>> n = np.array([0, 1, 2])
        >>> x_e, x_o, n_ext = SignalProcessor.even_odd_decomp(x, n)
        >>> x_e
        array([1., 3.])
        >>> x_o
        array([0., 2.])
        >>> n_ext
        array([0, 1, 2])
        """
        if np.any(np.iscomplex(x)):  # Check if input signal is complex
            raise ValueError("x is not a real sequence")
        
        x_flipped = np.flip(x)
        n_flipped = -np.flip(n)

        # Align signals
        x_ext, x_flipped_ext, n_ext = SignalProcessor.alignsigs(x, n, x_flipped, n_flipped)

        # Compute even and odd components
        x_e = 0.5 * (x_ext + x_flipped_ext)  # Even part
        x_o = 0.5 * (x_ext - x_flipped_ext)  # Odd part

        return x_e, x_o, n_ext

    @staticmethod
    def plot_signal(n: np.ndarray, x: np.ndarray, title: str = 'Signal', xlabel: str = 'n', ylabel: str = 'Amplitude', linewidth: int = 3, markersize: int = 10, xticks: np.ndarray = None, yticks: np.ndarray = None, show: bool = False) -> None:
        """
        Plots a discrete-time signal using a stem plot.

        Parameters:
        - n: np.ndarray, the range of indices.
        - x: np.ndarray, the signal values.
        - title: str, the title of the plot.
        - xlabel: str, the x-axis label.
        - ylabel: str, the y-axis label.
        - linewidth: int, the width of the stem lines.
        - markersize: int, the size of the markers.
        - xticks: np.ndarray, the x-axis tick values.
        - yticks: np.ndarray, the y-axis tick values.
        - show: bool, whether to display the plot yet or not.

        Example:
        >>> n = np.arange(-5, 6)
        >>> x = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        >>> SignalProcessor.plot_signal(n, x, title='Unit Impulse Signal', xlabel='n', ylabel='Amplitude', show=True)
        """
        plt.figure(figsize=(10, 4))
        
        markerline, stemlines, baseline = plt.stem(n, x)
        plt.setp(stemlines, 'linewidth', linewidth)
        plt.setp(markerline, 'markersize', markersize)
        
        plt.title(title)
        
        plt.xlabel(xlabel)
        if xticks is not None:
            plt.xticks(xticks)
        
        if yticks is not None:
            plt.yticks(yticks)
        plt.ylabel(ylabel)
        
        plt.grid()

        if show:
            plt.show()

    @staticmethod
    def subplot_signals(*s: tuple[np.ndarray, np.ndarray], rows: int = 1, cols: int = 1, **options: dict[str, any]) -> None:
        """
        Plots multiple discrete-time signals in subplots.

        Parameters:
        - s: tuple, a sequence of tuples containing the range of indices and signal values.
        - rows: int, the number of rows in the subplot grid.
        - cols: int, the number of columns in the subplot grid.
        - options: dict, additional options for customizing the plots.

        Example:
        >>> n1 = np.arange(-5, 6)
        >>> x1 = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        >>> n2 = np.arange(-5, 6)
        >>> x2 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        >>> SignalProcessor.subplot_signals((n1, x1), (n2, x2), rows=1, cols=2, title_1='Unit Impulse Signal', title_2='Unit Step Signal', xlabel_1='n', xlabel_2='n', ylabel_1='Amplitude', ylabel_2='Amplitude', linewidth=3, markersize=10, xticks_1=None, xticks_2=None, yticks_1=None, yticks_2=None)
        """
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5))
        axes = axes.flatten()

        for i, (n, x) in enumerate(s):
            markerline, stemlines, baseline = axes[i].stem(n, x)
            plt.setp(stemlines, 'linewidth', options.get('linewidth', 3))
            plt.setp(markerline, 'markersize', options.get('markersize', 10))
            
            if options.get(f'baseline_color_{i+1}') is not None:
                plt.setp(baseline, 'color', options.get(f'baseline_color_{i+1}'))
            
            axes[i].set_title(options.get(f'title_{i+1}', 'Signal'))
            
            axes[i].set_xlabel(options.get(f'xlabel_{i+1}', 'n'))
            if options.get(f'xticks_{i+1}') is not None:
                axes[i].set_xticks(options.get(f'xticks_{i+1}'))

            axes[i].set_ylabel(options.get(f'ylabel_{i+1}', 'Amplitude'))
            if options.get(f'yticks_{i+1}') is not None:
                axes[i].set_yticks(options.get(f'yticks_{i+1}'))
            
            axes[i].grid()

        plt.tight_layout()
        plt.show()
