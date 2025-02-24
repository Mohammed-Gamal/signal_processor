# 🌟 Signal Processor

A Python package for generating and processing discrete-time signals, including:
- Impulse sequence
- Step sequence
- Signal addition
- Signal multiplication
- Signal shifting
- Signal folding
- Even-odd decomposition
- Plotting signals

## 📘 Installation

```bash
pip install signal-processor
```

## 🕹️ Usage

```python
import numpy as np
from signal_processor import SignalProcessor

# Generate a unit impulse sequence
x, n = SignalProcessor.impseq(0, -5, 5)
SignalProcessor.plot_signal(n, x, title='Unit Impulse Signal', show=True)
```

## 📂 Project Structure

```bash
signal_processor/
├── signal_processor/
│   ├── __init__.py
│   └── processor.py
├── README.md
├── LICENSE
└── setup.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
