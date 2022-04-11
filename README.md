# Speech-Recognition-using-Brain-Inspired-HDC
Sean Lane's Master's Thesis Villanova University

Main Dataset: https://www.kaggle.com/datasets/jbuchner/synthetic-speech-commands-dataset - subset used within repo

Digit Implementation:
  MATLAB
  Run preprocessing code first 
    Ensure #bins and classes used in both preprocessing and main code are the same.
      If changing #bins, must edit lines 10, 400, and 401 
      If changing #dimensions, must edit lines 11, 345, and 401
      
Level Implementation:
  MATLAB
  Run preprocessing code first 
    Ensure #bins and classes used in both preprocessing and main code are the same.
      If changing #bins, must edit lines 10 and 218 
        must evaluate bin distribution for #level hypervectors
      If changing #dimensions, must edit lines 11, 109, 164, and 218
        If using modified correlation, must edit lines 174-180 for flipping bits
        
  VHDL
    Sim: audioHDC process to add variable waves to waveform


Acknowledgement: I would like to thank Dr. Jiao for his support and advice throughout this entire process and for introducing me to the topic of Hyperdimensional Computing in ECE 8405 during the spring of 2021. I would also like to thank Dongning Ma for his technical expertise and advice surrounding the topic of Hyperdimensional Computing. I would like to thank the entire ECE department for their help during my thesis and over my entire 5 years at Villanova.
