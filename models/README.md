# Model Weights

This folder is used to store the trained YOLOv8 model weights required by the Bone Fracture Detection and Classification Web Application.

## Required Model Files

The application uses the following trained model files:

- `best.pt` – Main YOLOv8 fracture detection model.
- `best_type_classifier.pt` – Model used for fracture type classification.
- `best_body_part_classifier.pt` – Model used for body-part classification.

## Model Files

The trained `.pt` files are not included directly in this GitHub repository because of their large file size.

To run the application locally, download the required model weights and place them in this folder:

```text
models/
├── best.pt
├── best_type_classifier.pt
└── best_body_part_classifier.pt
