

import numpy as np
import matplotlib.pyplot as plt
import h5py

# for a single image
image = 2000
image_path = "./data/TrainData/img/image_"+str(image)+".h5"
mask_path = "./data/TrainData/mask/mask_"+str(image)+".h5"

Train_x = np.zeros((128,128,6))
Train_y = np.zeros((128,128,1))

with h5py.File(image_path, 'r') as hdf:
    data_image = np.array(hdf.get('img'))
    data_image[np.isnan(data_image)] = 0.0000001

    data_red = data_image[:,:,3]  #red band
    data_nir = data_image[:,:,7]  #near infrared band
    data_ndvi = np.divide((data_nir - data_red),np.add(data_nir, data_red)) #calculating ndvi

    Train_x[:,:,0] = data_red             #red band 
    Train_x[:,:,1] = data_image[:,:,2]    #green band
    Train_x[:,:,2] = data_image[:,:,1]    #blue band
    Train_x[:,:,3] = data_ndvi            #ndvi band
    Train_x[:,:,4] = data_image[:,:,12]   #slope band
    Train_x[:,:,5] = data_image[:,:,13]   #elevation band
    

    
    
try:
    with h5py.File(mask_path, 'r') as hdf:
        data_mask = np.array(hdf.get('mask'))
        data_mask[np.isnan(data_mask)] = 0.0000001
        Train_y[:,:,0] = data_mask
        
except Exception as e:
        print("Error:", str(e))
x_train = np.reshape(Train_x,(Train_x.shape[0]*Train_x.shape[1], Train_x.shape[2]))
y_train = np.reshape(Train_y,(Train_y.shape[0]*Train_y.shape[1], Train_y.shape[2]))

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42, verbose=3, n_jobs=-1)
rfc.fit(x_train, y_train)
image = 2000
image_path = "./data/TrainData/img/image_"+str(image)+".h5"
mask_path = "./data/TrainData/mask/mask_"+str(image)+".h5"

Test_x = np.zeros((128,128,6))
Test_y = np.zeros((128,128,1))

with h5py.File(image_path, 'r') as hdf:
    data_image = np.array(hdf.get('img'))
    plt.imshow(data_image[:,:,0:3])
    plt.show()
    data_image[np.isnan(data_image)] = 0.0000001

    data_red = data_image[:,:,3]  #red band
    data_nir = data_image[:,:,7]  #near infrared band
    data_ndvi = np.divide((data_nir - data_red),np.add(data_nir, data_red)) #calculating ndvi

    Test_x[:,:,0] = data_red             #red band 
    Test_x[:,:,1] = data_image[:,:,2]    #green band
    Test_x[:,:,2] = data_image[:,:,1]    #blue band
    Test_x[:,:,3] = data_ndvi            #ndvi band
    Test_x[:,:,4] = data_image[:,:,12]   #slope band
    Test_x[:,:,5] = data_image[:,:,13]   #elevation band
    

try:
    with h5py.File(mask_path, 'r') as hdf:
        print(hdf.keys())
        data_mask = np.array(hdf.get('mask'))
        # data_mask[np.isnan(data_mask)] = 0.0000001
        plt.imshow(data_mask)
        plt.show()
        Test_y[:,:,0] = data_mask
        
except Exception as e:
        print("Error:", str(e))

x_test = np.reshape(Test_x,(Test_x.shape[0]*Test_x.shape[1], Test_x.shape[2]))
y_test = np.reshape(Test_y,(Test_y.shape[0]*Test_y.shape[1], Test_y.shape[2]))

y_pred = rfc.predict(x_test)
y_pred.shape
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
