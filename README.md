# MRI-Super-Resolution
This project aims to enhance 1.5T MRI scans to 3T MRI scans using deep-learning methods. This can be broken down into two parts:
- Simulating the degradation of 3T HR scans into 1.5T LR scans.
- Using these degraded 1.5T LR scans to train a deep-learning model that predicts their 3T HR counterparts.
## Simulation
Due to the lack of paired 1.5T-3T data, we need to simulate 1.5T data to use in training. This can be done by manipulating the $k$-space representation of the 3T HR scans and applying a set of physics-based noise and contrast changes to the resultant image. 
## Super-Resolution
Yet to be implemented.