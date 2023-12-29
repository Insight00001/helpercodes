import os, sys
import random
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import numpy as np 
import pathlib
import tensorflow as tf
#print(os.path.dirname(sys.executable))


class Helper():
    def view_random_images(self,target_dir,class_target):
        target_folder = target_dir+class_target
        random_image = random.sample(os.listdir(target_folder),1)
        img = mpimg.imread(target_folder + '/' + random_image)
        plt.imshow(img)
        plt.title(class_target)
        plt.axis('off')
        return img
    
    def get_classnames(self,data_dir):
        data_dir= pathlib.Path(data_dir)
        class_name = np.array(sorted([item.name for item in data_dir.glob('*')]))
        return class_name
    
    def view_folder(self,data_dir):
        for dir_path,dir_name,filename in os.walk(data_dir):
            return(f"the path is {dir_path} and there are {len(dir_name)} folders and {len(filename)} images")
        

    def prepare_image(self,filename,img_shape=224):
        img = tf.io.read_file(filename)
        img = tf.image.decode_image(img,channels=3)
        img = tf.image.resize(img,[img_shape,img_shape])
        #img = tf.expand_dims(filename, axis=0)

        img=img/255. # type: ignore
        return img

    def plot_pred_image(self,model,filename,class_names):
        img = self.prepare_image(filename)
        self.pred = model.predict(tf.expand_dims(filename, axis=0)) # make prediction
        if len(self.pred[0])>1:
            pred_class = class_names[self.pred.argmax()]
        else:
            pred_class = class_names[int(tf.round(self.pred)[0][0])] # type: ignore

        plt.imshow(img)
        plt.title(f"Prediction{pred_class}")
        plt.axis('off')
        return img




