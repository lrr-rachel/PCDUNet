# Author: Ruirui Lin, University of Bristol, United Kingdom.                  
# Supervisor: Dr Pui Anantrasirichai                                                                 
# Email: gf19473@bristol.ac.uk                                                
                                                                        
#!/usr/bin/python
from PIL import Image
import os, sys

# path = "./datasets/DodgeHeatWavw_restored/"
path = './results/dodgeHeatWavw/model/'

dirs = os.listdir( path )
# width_size = 480
# height_size = 384

width_size = 640
height_size = 384

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((width_size,height_size), Image.ANTIALIAS)
            imResize.save(f + '.png', 'PNG', quality=90)

resize()

