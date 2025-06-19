Of course! Here's a polished `README.md` with a clear note that the project is **work in progress**.

---

# Kernel QDA

This repository contains code for **Kernel Quadratic Discriminant Analysis**, a statistical method for classifying samples.
The approach utilizes (Gaussian) kernel embeddings to construct likelihood-based statistic for multilabel classification.
The implementation adaptively selects the kernel bandwith and regularisation ridge.

> ⚠ **Work in progress:**
> This repository is under active development.
> The code may be incomplete, and the API may change frequently.
> Please check back for future updates.

---

## 📁 Repository Structure

```
.
|-- requirements.txt
|
|-- tests/
|   |-- classification.ipynb
|   |-- testrun.py
|   | out/
|       |-- images/
|       |-- data/
|-- src/
|   |-- classification/
|   |-- utils/
|   |-- modules/
```

---

## 🔹 Main Components

| -------------------------------------- | ------------------------------------------------------ |
| `src/TwoSampleTests/classification.py` | main kernel likelihood ratio classifaction             |
| -------------------------------------- | ------------------------------------------------------ |
---


## 🔹 Contributors

* **Author:** Leonardo Santoro

---

## 🔹 References

* [Santoro, Waghmare and Panaretos (2025) "From Two Sample Testing to Singular Gaussian Discrimination"](https://arxiv.org/abs/2505.04613)
* Santoro, Waghmare and Panaretos (2025) "Likelihood Ratio Tests via Kernel Embeddings, to appear
* Santoro, Waghmare and Panaretos (2025) "Kernel QDA by Gaussian Embeddings, to appear

---
