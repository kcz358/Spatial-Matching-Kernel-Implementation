# Spatial-Matching-Kernel-Implementation
Python implementation for the 2006 CVPR paper : [Beyond Bags of Features: Spatial Pyramid Matching](https://inc.ucsd.edu/mplab/users/marni/Igert/Lazebnik_06.pdf)

## Usage
To use this repo, you should download [Caltech101](https://data.caltech.edu/records/mzrjq-6wc02) dataset and then put in the folder. Then you can run the python program using the command:
```
python main.py -L=<level>
```
Where L represents the pyramid level you desire to test.

## Experiment results
Here are my experiment results:
| Pyramid Level | Testing Accuracy |
|---------------|-------------------|
| 0             | 60.07             |
| 1             | 65.17             |
| 2             | 66.83             |
| 3             | 66.25             |

## For NTU student
If you are also taking SC4061/CZ4003 and is working on 3D stereo vision quesion in Lab2, feel free to check my implementation in the jupyter notebook
