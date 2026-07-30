# Assembled Models

`model/` owns dataset-specific or project-specific assembled models.

Examples:

- VitalSense radar-to-vital baselines.
- VitalDB-to-VitalSense radar-to-token transfer mappers.
- VitalDB VQ-VAE assembled models and frozen-teacher utilities.
- Future dataset-specific prediction models.

Reusable low-level blocks belong in `blocks/`. Concrete encoder-decoder or
tokenizer frameworks belong in `codec/`; dataset-specific wrappers around those
frameworks belong here.

Every `.py` file in this folder requires a same-name `.md` document.
