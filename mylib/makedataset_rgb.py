# --- Tools and parameters ---
import os
import numpy as np
import cv2
import glob
import random
import matplotlib.pyplot as plt
import shutil
from distutils import dir_util

class MakeDataSetRGB():
    '''
    This class generates augmented dataset especially for Deep Lerning(classification)
    Usage: 
    import mylib.makedataset_rgb as mkdataset
    make_dataset = mkdataset.MakeDataSetRGB()
    make_dataset.do_augmentation(dataset_folder_name = "dataset")
    '''
    def __init__(self, do_reverse=True, 
                 do_gamma_correction=True, 
                 do_add_noise=True, 
                 do_cut_out=True, 
                 do_deformation=True, 
                 irate=1,
                 img_size = [224,224]):#[img_width, img_hight]
        self.do_reverse =do_reverse
        self.do_gamma_correction = do_gamma_correction
        self.do_add_noise = do_add_noise
        self.do_cut_out = do_cut_out
        self.do_deformation = do_deformation
        self.irate = irate # inflation rate for deformation
        self.img_size = img_size
        self.save_rgb2gray = False
        self.class_folder = "1"

        pathc = os.getcwd()
        self.pathc = pathc.replace("\\", '/')
        self.pathc = self.pathc.replace("/mylib", "")
        #print(self.pathc)

        path_mrcnn_dataset = "path_mrcnn_dataset"#You should change here

        self.img_input_folder = pathc + '/dataset/'+self.class_folder
        self.img_output_folder = '/dataset_aug'

        self.imgs = glob.glob(self.img_input_folder + '/*')
        
        self.n = 0
        
    def imgSave(self, img):
        self.n += 1
        filename1 = self.img_output_folder+"/img_" +str(self.n).zfill(5)+'.png'
        #f.write( path_mrcnn_dataset  + file_name_top + str(n).zfill(4) + '.png'+'\n' )
        size = (self.img_size[1], self.img_size[0])
        img = cv2.resize(img, size)
        cv2.imwrite(filename1, img)
        if self.save_rgb2gray == True:
            cv2.imwrite(filename5, cv2.cvtColor(img.astype('uint8'),cv2.COLOR_RGB2GRAY))

    def horizontalFlip(self,img):
        img = img[:,::-1,:]
        return img

    def luminanceUp(self,img):
        dst = img*1.3
        return dst
    def luminanceDown(self,img):
        dst = img*0.7
        return dst

    def addNoise(self,img):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        row,col,ch = img.shape
        if np.random.uniform(0, 1) > 0.7:
            # white noise
            pts_x = np.random.randint(0, col-1 , 70)
            pts_y = np.random.randint(0, row-1 , 70)
            if row < 300:
                pts_x = np.random.randint(0, col-1 , 50)
                pts_y = np.random.randint(0, row-1 , 50)
            img[(pts_y,pts_x)] = (255,255,255)

            # black noise
            pts_x = np.random.randint(0, col-1 , 70)
            pts_y = np.random.randint(0, row-1 , 70)
            if row < 300:
                pts_x = np.random.randint(0, col-1 , 50) 
                pts_y = np.random.randint(0, row-1 , 50)
            img[(pts_y,pts_x)] = (0,0,0)

            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        else:
            pts_x = np.random.randint(0, col-1 , 20)
            pts_y = np.random.randint(0, row-1 , 20)
            img[(pts_y,pts_x)] = (255,255,255)
            pts_x = np.random.randint(0, col-1 , 20)
            pts_y = np.random.randint(0, row-1 , 20)
            img[(pts_y,pts_x)] = (0,0,0)
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        return img

    def cutOut(self,img):
        rows,cols,ch = img.shape
        #rows is height, cols is width
        imgco = img
        rn1 = random.randint(0, int(cols))#In python3, these should be int() 
        rn2 = random.randint(0, int(rows))
        w = random.randint(0, int(cols/5))
        h = random.randint(0, int(rows/5))
        cval = random.randint(0, 255)
        cv2.rectangle(imgco, (rn1, rn2), (rn1 + w, rn2 + h), (cval, cval, cval), -1)
        self.imgSave(imgco)
        #return imgco

    def homographyTrans(self,img):
        rows,cols,ch = img.shape
        rn1 = random.randint(0, int(cols/10))
        rn2 = random.randint(0, int(cols/10))
        rn3 = random.randint(int(cols*9/10), cols)
        rn4 = random.randint(int(cols*9/10), cols)
        rn5 = random.randint(0, int(rows/10))
        rn6 = random.randint(0, int(rows/10))
        rn7 = random.randint(int(rows*9/10), rows)
        rn8 = random.randint(int(rows*9/10), rows)

        pts1 = np.float32([[rn1,rn5],[rn3,rn6],[rn2,rn7],[rn4,rn8]])
        pts2 = np.float32([[0,0],[cols,0],[0,rows],[cols,rows]])

        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(img,M,(cols,rows))

        self.imgSave(dst)

    def do_augmentation(self, dataset_folder_name):
        path_dataset = self.pathc + "/" + dataset_folder_name #dataset_folder_name = "dataset" or "dataset_val"
        for i in os.listdir(path_dataset):
            print("Now executing augmentation :" + dataset_folder_name +"/"+ str(i))
            self.n = 0
            self.class_folder = i
            self.img_input_folder = path_dataset + "/" + self.class_folder
            self.img_output_folder = path_dataset + "_aug/" + self.class_folder
            if os.path.exists(self.img_output_folder) == False:
                os.makedirs(self.img_output_folder)
            cnt = 0
            #img_input_folder = self.img_input_folder
            self.imgs = glob.glob(self.img_input_folder + '/*')

            # --- Save loaded images ---
            for cnt in range(len(self.imgs)):
                self.img_input_folder = path_dataset + '/'+self.class_folder
                self.img_output_folder = path_dataset + '_aug/'+self.class_folder
                img_input_folder = self.img_input_folder
                self.imgs = glob.glob(img_input_folder + '/*')
                image = cv2.imread(self.imgs[cnt])
                self.imgSave(image)
                cnt = cnt + 1

            # --- Save reversed images ---
            if self.do_reverse == True:
                cnt = 0
                img_input_folder = self.img_output_folder# NOT self.img_input_folder
                imgs = glob.glob(img_input_folder + '/*')
                for cnt in range(len(imgs)):
                    image = cv2.imread(imgs[cnt])
                    image_rev = self.horizontalFlip(image)
                    self.imgSave(image_rev)
                    cnt = cnt + 1

            # --- Save gamma corrected images ---
            if self.do_gamma_correction == True:
                cnt = 0
                img_input_folder = self.img_output_folder# NOT self.img_input_folder
                imgs = glob.glob(img_input_folder + '/*')
                for cnt in range(len(imgs)):
                    image = cv2.imread(imgs[cnt])
                    image_lumup = self.luminanceUp(image)
                    image_lumdown = self.luminanceDown(image)

                    self.imgSave(image_lumup)
                    self.imgSave(image_lumdown)
                    cnt = cnt + 1

            # --- Save noised images ---
            if self.do_add_noise == True:
                cnt = 0
                img_input_folder = self.img_output_folder# NOT self.img_input_folder
                imgs = glob.glob(img_input_folder + '/*')
                for cnt in range(len(imgs)):
                    image = cv2.imread(imgs[cnt])
                    image_pn = self.addNoise(image)
                    self.imgSave(image_pn)

            # --- Save cut-out images ---
            if self.do_cut_out == True:
                cnt = 0
                img_input_folder = self.img_output_folder# NOT self.img_input_folder
                imgs = glob.glob(img_input_folder + '/*')
                for cnt in range(len(imgs)):
                    image = cv2.imread(imgs[cnt])
                    self.cutOut(image)

            # --- Save deformed images ---
            if self.do_deformation == True:
                cnt = 0
                img_input_folder = self.img_output_folder# NOT self.img_input_folder
                imgs = glob.glob(img_input_folder + '/*')
                for cnt in range(len(imgs)):
                    image = cv2.imread(imgs[cnt])

                    for i in range(self.irate):
                        self.homographyTrans(image)
            #self.f.close()