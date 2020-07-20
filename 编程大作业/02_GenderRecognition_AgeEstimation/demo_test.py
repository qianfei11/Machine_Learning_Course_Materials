import numpy as np
from PIL import Image
#import caffe
import os
def AgeEstimation(image):
    ##########You need to fill out this section###########
    ##########THIS SECTION CONTAINS PREDICT STEPS#####################
    
    return age   

def GenderIdentification(image):   
    ##########You need to fill out this section###########
    ##########THIS SECTION CONTAINS PREDICT STEPS#####################   
      
    return gender

if __name__ == '__main__':
    
    ############Section 1.Steps of Running environment configuration#############
    ###########If you need, fill it;otherwise, skip it###############

    ############End of section 1###########
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('%s%s' % (filepath, allDir))
        for filename in os.listdir(child):
            img = os.path.join('%s/%s' % (child, filename))

            ############Section 2.Steps of Image Preprocessing#############
            ###########If you need, fill it;otherwise, skip it###############
            
            ############End of section 2###########
            
            ############Section 3.Steps of Prediction#############
            ###############Generally you do not need to revise###############
            age = AgeEstimation(image)
            gender = GenderIdentification(image)
            ############End of section 3###########
            
            fobj.write(filename)
            fobj.write(' ')
            fobj.write(age)
            fobj.write(' ')
            fobj.write(gender)
            fobj.write('\n')
    fobj.close()
        



