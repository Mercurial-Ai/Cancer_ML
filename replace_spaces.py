import os
base = os.getcwd()
for patient in os.listdir("."):
    patient = os.path.join(base, patient)
    for dir in os.listdir(patient):
        dir = os.path.join(patient, dir)
        for f in os.listdir(dir):
            r = f.replace(" ", "_")
            if(r != f):
                os.rename(f,r)
