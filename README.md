# EC523 Project
___
This is based on [suinleelab/derm_audit](https://github.com/suinleelab/derm_audit).

## Instructions for Running

### Original Repository

#### Dependencies and Environment
___
The repository comes with a `environment.yaml` file. You can create an python environment based on this using `conda env create -f environment.yaml`. 
However, the `environment.yaml` currently only works on Linux as it specifies some Linux-specific packages. 
- Setting this up on the SCC, we are able to create the environment via this method, but it did output some error.
- We ultimately got the environment to run, so we just ignored the error
- We still had to pip install `onnx2pytorch` and `geffnet`

If you want to manually create an environemnt (on Windows for example), we were able to get the code working with Python 3.12 and pip installing the 
`pandas`, `torchvision`, `torch`, `tqdm`, `protobuf`, `onnx`, `onnx2pytorch`, `geffnet`, `tensorboard` packages.

#### Models and Datasets
____
The scripts allows the users to test the following classifiers: `DeepDerm`, `ModelDerm`, `Scanoma`, `SSCD`, and `SIIMISIC`. 
- The paper links the following classifiers: [DeepDerm](https://zenodo.org/records/6784279#.ZFrDc9LMK-Z), [ModelDerm](https://figshare.com/articles/code/Caffemodel_files_and_Python_Examples/5406223), 
and [SIIMISIC](https://zenodo.org/records/10049217).
- The naming schemes and URLs for ModelDerm and SIIMISIC are confusing to us at the moment, and we only plan to work with DeepDerm regardless. 

The paper also links their GAN models [here](https://zenodo.org/records/10049217). This is the same link as the SIIMISIC. 
- I'm not sure what the first 3 files with the strange coded names are.
- The rest of the file names are structered as `<classifier>_<dataset>.pth`.
- To be clear, these are GAN models, not classifiers. I believe the `<classifier>` field just indicates which classifier was used to help train the GAN and the `<dataset>` field is the dataset they were trained on.

The paper links the relevant datasets: [ISIC-2019](https://challenge.isic-archive.com/data/#2019), [Fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k), and [DDI](https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965). 
- I haven't figured out how to use the Fitzpatrick images as they do not provide the actual image files, but links to the images instead. 
- However, you can just directly download the ISIC and DDI images to your device. 
- I've only tested ISIC so far, and the scripts require both the `ISIC_2019_Training_GroundTruth.csv` and `ISIC_2019_Training_Input` image set to be downloaded and in the same directory.

#### Set-up Commands
___
The repository comes with a `prepare.sh` shell script to configure some files and directories. 

- Modify the path variables in `prepare.sh` to your classifers and your datasets.
  - Specifically for ISIC, make sure the directory you link contains both the `ISIC_2019_Training_GroundTruth.csv` file and the `ISIC_2019_Training_Input` directory.
- Make sure you can run bash commands (Git Bash, WSL, Cygwin all let you run bash commands this on Windows).
- Make sure `wget` is a runnable bash command. I installed this with Chocolatey on Windows via `choco install wget -y`, but I'm sure there are other methods.
- Make sure `protoc` is a runnable command. You can test this with `protoc --version`. To install for Windows:
  - You need to download and extract the appropriate `.zip` for your platform from the [release pages](https://github.com/protocolbuffers/protobuf/releases)
  - Locate the `protoc.exe` file.
  - Add the path to the directory containing `protoc.exe` file to your system PATH environment variable.
 
Much of these instructions will be different on a Linux or Mac environment. For the SCC, it was not necessary to install `wget` and `protoc` as it was already there.

#### Scripts
___

`evaluate_classifier.py` just gives the accuracy of a given classifier model on a given dataset.
- You need to specify the classifier and the dataset in the command line arguments (or just hardcode them in the file)

`train.py` trains the GAN model given a classifier and given a dataset.
- Only run this if you want to train a GAN model. This isn't really necessary since the paper provides the pre-trained GAN models.
- You will have to comment and uncomment lines depending on which dataset and classifer you want to use to train the GAN model.

`test.py` generates the counterfactual images.
- This script takes several command line arguments
  - `--checkpoint_path` is the path to your GAN model.
  - `--dataset` is the dataset.
  - `--classifier` is the classifier.
  - `--output` is the output directory for your counterfactuals.
  - `--max_images` specifies how many iamges to generate counterfactuals for.
  - `--batch_size` I'm not sure what this is, I just left the default value.
 
- The generated image files have the following format: `original image` | `beign counterfactual` | `malginant counterfactual`

#### Script Fixes
___
- All the relevant scripts imports the code for all the classifiers and dataset, even if that's not the one you are using. 
This isn't an issue except for ModelDerm, which requires a missing `modelderm_labels.py` file for some reason. 
As a result, I was only able to get the scripts working by commenting out every instance of `from models import ModelDermClassifier`.

- If you get the error `TypeError: Couldn't build proto file into descriptor pool: duplicate file name caffe.proto`, restarting the kernel fixes this error temporarily. 

### SCC Set-up


- Create a desktop session with the module `miniconda academic-ml/fall-2024`.
  ![image](https://github.com/user-attachments/assets/fa766e3c-1cdd-4faf-8be1-e8986b037a22)
- Clone the repository into your local, personal directory. Ex) `/projectnb/ec523kb/students/howellx`.
- To create the conda environment: 
  - Navigate to the repository, which should have an `environment.yaml` file.
  - Create the conda environment with `conda env create -f environment.yaml`. This may return an error, but it should still create the environment with most of the needed packages.
  - Activate the conda environment with `conda activate torch9`
  - I still needed to install a few packages: `pip install onnx2pytorch` and `pip install geffnet`
- Modify the file links `prepare.sh` file:
  - You only need to link the classifier and datasets, not the GANS.
  - We are only using the DeepDerm classifier, so the only link for classifiers we need to create is:
    - `ln -s /projectnb/ec523kb/projects/skin_tone/DDI-models/DDI-models/deepderm.pth ${CLASSIFIER_DIR}/`
  - For datasets, we would need to replace their ISIC line with: 
    - `ln -s /projectnb/ec523kb/projects/skin_tone/isic/ ${DATA_DIR}/isic`
  - We will also need to link Fitzpatrick17k when we get that.
  - All other links to other classifiers and datasets are unnecessary, and you can delete those linking commands.
  - The final file should look similar to:
    ![image](https://github.com/user-attachments/assets/bc361d8d-876b-4936-94e4-ae17bb981b1b)
- Run the file with `bash prepare.sh`
  - This should create some caffe files, which I'm not sure what they do but they are necessary for the scripts to load in the models correctly.  
  - This should create a `data` and `pretrained_classifiers` directory. Within the directories, there should be links to the datasets and models.
    ![image](https://github.com/user-attachments/assets/cc277b04-a2b4-4d39-925d-3f2f87e13303) <br> <br>
    ![image](https://github.com/user-attachments/assets/5c180592-c1f8-46a8-8edd-ee2bff01b669)
- Activate the conda environment with `conda activate torch9`
- You should be able to run the scripts now. Ex) `python test.py`


