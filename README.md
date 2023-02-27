# QSM reconstruction using Total Generalized Variation (TGV-QSM)

## References

Kristian Bredies and Christian Langkammer:

- Bredies, K., Ropele, S., Poser, B. A., Barth, M., & Langkammer, C. (2014). Single-step quantitative susceptibility mapping using total generalized variation and 3D EPI. In Proceedings of the 22nd Annual Meeting ISMRM (Vol. 62, p. 604).

- Langkammer, C., Bredies, K., Poser, B. A., Barth, M., Reishofer, G., Fan, A. P., ... & Ropele, S. (2015). Fast quantitative susceptibility mapping using 3D EPI and total generalized variation. Neuroimage, 111, 622-630.

## Install and run

```bash
# install miniconda with dependencies
wget https://repo.anaconda.com/miniconda/Miniconda2-4.6.14-Linux-x86_64.sh
bash Miniconda2-4.6.14-Linux-x86_64.sh -b -p miniconda2
miniconda2/bin/conda install -y -c anaconda cython==0.29.4
miniconda2/bin/conda install -y numpy
miniconda2/bin/conda install -y pyparsing
miniconda2/bin/pip install scipy==0.17.1 nibabel==2.1.0
miniconda2/bin/pip install --upgrade cython

# install TGV_QSM
git clone https://github.com/QSMxT/TGV_QSM.git
cd "TGV_QSM"
../miniconda2/bin/python setup.py install

# run TGV_QSM
miniconda2/bin/tgv_qsm
```

## Usage

```
tgv_qsm [-h] -p PHASE -m MASK [-o OUTPUT_SUFFIX]
        [--alpha ALPHA ALPHA | --factors FACTORS [FACTORS ...]]
        [-e EROSIONS] [-i ITERATIONS [ITERATIONS ...]]
        [-f FIELDSTRENGTH] [-t ECHOTIME] [-s] [--ignore-orientation]
        [--save-laplacian] [--output-physical] [--no-resampling]
        [--vis] [-v]
```

Note:

- -t TE in seconds
- -f fieldstrength in Tesla
- -s autoscaling for SIEMENS phase data

