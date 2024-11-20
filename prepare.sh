#!/bin/bash

# Image datasets
DATA_DIR="data"

# Pretrained classifier checkpoints
CLASSIFIER_DIR="pretrained_classifiers"

# Model code
MODEL_DIR="models"

if [ ! -d ${DATA_DIR} ]; then
    mkdir ${DATA_DIR} 
fi

if [ ! -d ${CLASSIFIER_DIR} ]; then
    mkdir ${CLASSIFIER_DIR} 
fi

# Link in classifiers
### you will need to edit these paths to match the locations of your model weight files ###
# ln -s /projects/leelab3/derm/models/DDI/DDI-models/deepderm.pth ${CLASSIFIER_DIR}/
# ln -s /projects/leelab3/derm/models/tflite/scanoma.onnx ${CLASSIFIER_DIR}/
# ln -s /projects/leelab3/derm/models/tflite/sscd.onnx ${CLASSIFIER_DIR}/
# ln -s /projects/leelab3/derm/models/modelderm_2018/70616.caffemodel ${CLASSIFIER_DIR}/
# ln -s /projects/leelab3/derm/models/modelderm_2018/deploy.prototxt ${CLASSIFIER_DIR}/
# ln -s /projects/leelab3/derm/models/modelderm_2018/mean224x224.binaryproto ${CLASSIFIER_DIR}/
# ln -s /projects/leelab3/derm/models/modelderm_2018/modelderm_labels.py ${MODEL_DIR}/

# ln -s ~/OneDrive/Documents/school/EC523/derm_audit/models/10049217/deepderm_isic.pth ${CLASSIFIER_DIR}/
# ln -s ~/OneDrive/Documents/school/EC523/derm_audit/models/10049217/deepderm_f17k.pth ${CLASSIFIER_DIR}/
# ln -s ~/OneDrive/Documents/school/EC523/derm_audit/models/10049217/scanoma_isic.pth ${CLASSIFIER_DIR}/
# ln -s ~/OneDrive/Documents/school/EC523/derm_audit/models/10049217/scanoma_f17k.pth ${CLASSIFIER_DIR}/
# ln -s ~/OneDrive/Documents/school/EC523/derm_audit/models/10049217/modelderm_isic.pth ${CLASSIFIER_DIR}/
# ln -s ~/OneDrive/Documents/school/EC523/derm_audit/models/10049217/modelderm_f17k.pth ${CLASSIFIER_DIR}/
# ln -s ~/OneDrive/Documents/school/EC523/derm_audit/models/10049217/siimisic_isic.pth ${CLASSIFIER_DIR}/
# ln -s ~/OneDrive/Documents/school/EC523/derm_audit/models/10049217/siimisic_f17k.pth ${CLASSIFIER_DIR}/
# ln -s ~/OneDrive/Documents/school/EC523/derm_audit/models/10049217/sscd_isic.pth ${CLASSIFIER_DIR}/
# ln -s ~/OneDrive/Documents/school/EC523/derm_audit/models/10049217/sscd_f17k.pth ${CLASSIFIER_DIR}/

# ln -s ~/OneDrive/Documents/school/EC523/derm_audit/models/6784279/DDI-models/DDI-models/deepderm.pth ${CLASSIFIER_DIR}/

ln -s /projectnb/ec523kb/projects/skin_tone/DDI-models/DDI-models/deepderm.pth ${CLASSIFIER_DIR}/


# ln -s ~/OneDrive/Documents/school/EC523/derm_audit/models/10049217/9c_b7ns_1e_224_ext_15ep_best_fold0.pth ${CLASSIFIER_DIR}/
# ln -s ~/OneDrive/Documents/school/EC523/derm_audit/models/10049217/9c_b6ns_1e_224_ext_15ep_best_fold1.pth ${CLASSIFIER_DIR}/
# ln -s ~/OneDrive/Documents/school/EC523/derm_audit/models/10049217/9c_b5ns_1e_224_ext_15ep_best_fold2.pth ${CLASSIFIER_DIR}/

# Link in datasets
### you will also need to edit these paths to match the locations of your datasets ###
# ln -s ~/OneDrive/Documents/school/EC523/derm_audit/datasets/fitzpatrick ${DATA_DIR}/f17k
ln -s /projectnb/ec523kb/projects/skin_tone/isic/ ${DATA_DIR}/isic
# ln -s ~/OneDrive/Documents/school/EC523/derm_audit/datasets/ddidiversedermatologyimages ${DATA_DIR}/ddi

# Download caffe proto; at time of writing, PyCaffe isn't compatible with 
# the necessary version of PyTorch
wget -P ${CLASSIFIER_DIR}/ https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto 
# produce caffe.pb2
protoc --proto_path ${CLASSIFIER_DIR} --python_out ${MODEL_DIR} ${CLASSIFIER_DIR}/caffe.proto
