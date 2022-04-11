# Speech-Recognition-using-Brain-Inspired-HDC
Sean Lane's Master's Thesis Villanova University

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
