import os
from dateutil import parser


#path = "C:/Users/vince/Documents/animations/Atwood_machine/Atwood_om_dr_4.415_6.010_frames"
#path = "C:/Users/vince/Documents/animations/Atwood_machine/Atwood_om_dr_4.419_6.014_frames"
#path = "C:/Users/vince/Documents/Animations_Videos/Pendule_Elastique_Video/X_Y_50_100_frames/"

fileName = "Pdl_Horizontal"

#myPath = "C:/Users/vince/PycharmProjects/Project_Perso/Simulations/figures/"
myPath = "C:/Users/vince/OneDrive - UCL/Simulations/figures/" + fileName

#files = sorted(os.listdir(myPath), key=os.path.getmtime)
files = sorted(os.listdir(myPath), key=lambda x: os.path.getmtime(os.path.join(myPath, x)))

i = 1

for file in files:
    #if file[0] == 'f':
    #    src = os.path.join(myPath, file)
    #    dst = os.path.join(myPath, "frame" + '{:07d}'.format(1604+5*i) + ".png")
    #    os.rename(src, dst )

    src = os.path.join(myPath, file)
    #end = file.split("_")[1]

    #dst = os.path.join(myPath, "frame" + '{:07d}'.format(i) + ".png")
    #dst = os.path.join(myPath, fileName + end)
    dst = os.path.join(myPath, fileName + '_{:d}'.format(i) + ".png")

    #os.rename(src, dst)
    i += 1
