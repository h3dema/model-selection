# Probabilistic Model Confidence Framework

This code defines a flexible and extensible framework for generating and evaluating models with probabilistic confidence scores.
Each model is characterized by attributes such as energy, size, and confidence (per class).
We also incorporate uncertainty through class-specific or uniform standard deviations so the model can show different levels of uncertainty per class per prediction.


In many machine learning and simulation contexts, models can be evaluated by their confidence scores.
This framework introduces a probabilistic approach for simulating model prediction behavior.

- Models can have **different levels of uncertainty** per class.
- Confidence is **sampled from a normal distribution** and normalized to reflect how likely a model is to produce reliable predictions for a given class.
- This allows for **more realistic modeling** of noisy or uncertain systems.


### `Model` Class

Represents a single model with the following attributes:

- `energy`: A float > 0 representing the model's energy.
- `size`: A float > 0 representing the model's size.
- `confidence_base`: The base confidence score (between 0 and 1).
- `stdev`: Either a single float (uniform across classes) or a list of floats (one per class).
- `classes`: A list of class labels (e.g., `[1, 2, 3, 4]`).

#### `confidence(c: int) → float`

Returns a normalized confidence score for class `c`, which reflects how likely the model is to produce a confident prediction for class `c`, given its uncertainty.
