To perform inference using Meta's Segment Anything Model 2.1 (SAM 2.1) in a Python notebook, follow these steps:

1. **Clone the SAM 2.1 Repository:**
   ```python
   !git clone https://github.com/facebookresearch/sam2.git
   ```
   This command clones the SAM 2.1 repository to your local environment.

2. **Navigate to the SAM 2.1 Directory:**
   ```python
   %cd sam2
   ```
   Ensure you're in the directory containing `setup.py`.

3. **Install SAM 2.1 in Development Mode:**
   ```python
   !pip install -e .[dev] -q
   ```
   This installs SAM 2.1 along with its development dependencies.

4. **Download Model Checkpoints:**
   ```python
   %cd checkpoints
   !./download_ckpts.sh
   ```
   This script downloads the necessary model checkpoints.

5. **Install the `supervision` Library:**
   ```python
   !pip install supervision -q
   ```
   The `supervision` library aids in visualizing segmentation results.

6. **Import Required Libraries:**
   ```python
   import torch
   from sam2.build_sam import build_sam2
   from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
   import supervision as sv
   from PIL import Image
   import numpy as np
   ```

7. **Set Up the Model and Mask Generator:**
   ```python
   model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
   checkpoint = "path_to_checkpoint.pt"  # Replace with the actual path
   sam2 = build_sam2(model_cfg, checkpoint, device="cuda")
   mask_generator = SAM2AutomaticMaskGenerator(sam2)
   ```
   Replace `"path_to_checkpoint.pt"` with the path to your downloaded checkpoint.

8. **Load and Process the Input Image:**
   ```python
   image_path = "path_to_image.jpg"  # Replace with your image path
   image = Image.open(image_path)
   image_np = np.array(image)
   ```
   Ensure the image is in RGB format.

9. **Generate Segmentation Masks:**
   ```python
   result = mask_generator.generate(image_np)
   detections = sv.Detections.from_sam(sam_result=result)
   mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
   annotated_image = image_np.copy()
   annotated_image = mask_annotator.annotate(annotated_image, detections=detections)
   ```

10. **Visualize the Results:**
    ```python
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(annotated_image)
    axes[0].set_title('Segmented Image')
    axes[0].axis('off')

    axes[1].imshow(image_np)
    axes[1].set_title('Original Image')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
    ```
    This code displays the original and segmented images side by side.

For more detailed examples and advanced usage, refer to the [SAM 2.1 GitHub repository](https://github.com/facebookresearch/sam2). 
