# EC523 Project

## SCC Set-up

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


