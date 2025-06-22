# Image Filter Generator

## How to Run

You'll need to install the dependencies first. You may use Anaconda or Python's built-in virtual environments. The project was implemented using Python 3.10, it is recommended you use the same version.
Once you activate your virtual environment, install the packages listed in the `requirements.txt` file.

Then, run `python gui.py` to launch the GUI application. Select the desired input image, model corresponding to the filter you want to apply, and give an output path. For the best results, leave the "Tiled inference" option on. Press "Apply" to apply the filter. This may take a while, the output image will appear on the application when it is ready.

## How to Train

If you want to use Google Colab for training, the training code is much well-suited for it. It is also fine if you want to run it locally. For both cases, you just need to modify the `INPUT_IMAGES_PATH`, `GROUND_TRUTH_IMAGES_PATH`, `FILTER_NAME` and `MODEL_SAVE_DIR` variables in the code appropriately. If you are not using Google Colab, remove the two lines at the top:

```python
from google.colab import drive
drive.mount('/content/drive')
```

The training code expects an input image and its corresponding ground truth image to have the same name under their respective directories.

## How to Generate Filtered Datasets

This document explains how to use the generate\_filter\_dataset\_combined.py script to create new, filtered datasets for training.

**Overview**

This script applies a selected programmatic filter (e.g., Oil Painting, Pencil Sketch) to a directory of source images and saves the results to a new directory. It is designed to run both on a local machine and in a Google Colab environment.

#### **Prerequisites**

Ensure you have the necessary Python libraries installed. You can install them using pip:

pip install opencv-python numpy tqdm

#### **Configuration**

Before running the script, you must configure the execution environment by editing this line at the top of the file:

```bash
# Set this to True if running on Google Colab, False if running locally  
RUNNING_IN_COLAB = True 
```

* Set to True if you are running the script in a Google Colab notebook. The script will expect your data to be in Google Drive and will attempt to mount it.  
* Set to False if you are running the script on your local computer.

#### **Usage**

The script is run from the command line (or a Colab code cell) and accepts several arguments to control its behavior.

**General Format:**

```bash
python generate_filter_dataset_combined.py --filter <filter_name> [options]
```

#### **Examples**

**1\. Applying an Oil Painting Filter (in Google Colab)**

This command will use the default paths configured in the script, apply the oil painting filter, and save the results to a new directory named oil\_filtered.

```bash
!python generate_filter_dataset_combined.py --filter oil
```

**2\. Applying a Pencil Sketch Filter (Locally)**

This command specifies local paths, applies the pencil sketch filter, and saves the results to a specific output folder.

```bash
python generate_filter_dataset_combined.py --filter sketch --base_path "C:/MyProject/datasets/adobe5k" --output_subdir "sketch_outputs"
```

**3\. Applying a Cartoon Filter with Custom Parameters**

This command applies the cartoon filter but overrides the default parameters to create a different style.

```bash
python generate_filter_dataset_combined.py --filter cartoon --cartoon_d 5 --cartoon_sigma_color 150
```

#### **Command-Line Arguments Explained**

**Required Argument:**

* \--filter: The type of filter to apply.  
  * **Choices:** oil, sketch, grayscale, sepia, cartoon

**Path Arguments:**

* \--base\_path: The root directory for your datasets.  
  * *On Colab*, this defaults to your Google Drive path.  
  * *Locally*, you **must** provide this path if it differs from the placeholder.  
* \--input\_subdir: The subdirectory within base\_path that contains the raw source images. (Default: raw)  
* \--output\_subdir: The subdirectory within base\_path where the filtered images will be saved. If not provided, it defaults to \<filter\_name\>\_filtered (e.g., oil\_filtered).

**Filter-Specific Arguments (Optional):**

You can tune the effect of each filter by providing these optional arguments.

* **Oil Painting (--filter oil)**  
  * \--oil\_size: Size of the pixel neighborhood. (Default: 10\)  
  * \--oil\_dyn\_ratio: Contrast parameter. (Default: 1\)  
* **Pencil Sketch (--filter sketch)**  
  * \--sketch\_sigma\_s: Controls the smoothing level. (Default: 60\)  
  * \--sketch\_sigma\_r: Controls detail preservation. (Default: 0.07)  
  * \--sketch\_shade\_factor: Controls the brightness of the sketch. (Default: 0.05)  
* **Cartoon Effect (--filter cartoon)**  
  * \--cartoon\_d: Diameter of the pixel neighborhood for the bilateral filter. (Default: 9\)  
  * \--cartoon\_sigma\_color: Filter sigma in the color space. (Default: 250\)  
  * \--cartoon\_sigma\_space: Filter sigma in the coordinate space. (Default: 250\)  
  * \--cartoon\_block\_size: Block size for edge detection. (Default: 9\)  
  * \--cartoon\_c: Constant subtracted from the mean for edge detection. (Default: 2\)

