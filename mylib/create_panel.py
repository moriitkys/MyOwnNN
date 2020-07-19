import tkinter

class CreatePanel():
    '''
    This class creates UI panel for setting parameters especially for Deep Lerning
    Usage: 
    import mylib.create_panel as create_panel
    setting_panel = create_panel.CreatePanel()
    setting_panel.create_buttons()
    ~ ex1) getting a boolean param
    flag_train = setting_panel.flag_train
    ~ ex2) getting a float param (from radio buton)
    ratio_train = float(setting_panel.var_sp.get())#0.0 ~ 1.0
    ~
    Notes:
    If you change input image size, you should give arguments like bellow:
    setting_panel = create_panel.CreatePanel(img_size_mynet = [56,56])
    '''
    def __init__(self, img_size_resnet = [224,224],
                        img_size_mbnet = [192,192],
                        img_size_mynet = [28,28]):
        self.tki = tkinter.Tk()
        self.tki.geometry('350x400')
        self.tki.title('Settings')
        
        self.flag_train = False
        self.flag_aug = False
        self.flag_split = True
        self.type_backbone = "MyNet"
        self.layer_name_gradcam = " "
        self.img_sizes = [img_size_resnet, img_size_mbnet, img_size_mynet]
        self.img_size = self.img_sizes[2]
        
        self.var_sp = tkinter.StringVar()
        self.var_sp_epochs = tkinter.StringVar()

    def tkinter_callback(self, event):
        if event.widget["bg"] == "SystemButtonFace":
            event.widget["bg"] = "red"
        else:
            event.widget["bg"] = "SystemButtonFace"

    def click_flag_train(self,):
        if self.flag_train == True:
            self.label_var['text'] = 'Mode: Inference'
            self.flag_train = False
        else:
            self.label_var['text'] = 'Mode: Train'
            self.flag_train = True

    def click_start(self,):
        self.tki.destroy()

    def create_buttons(self):
        # Create labels and buttons

        self.radio_value_aug = tkinter.IntVar() 
        self.radio_value_split = tkinter.IntVar() 
        self.radio_value_net = tkinter.IntVar() 
        self.radio_value_aug.set(2)
        self.radio_value_split.set(1)
        self.radio_value_net.set(1)

        self.label_var = tkinter.Label(self.tki, text='Mode: Inference')
        self.label_var.place(x=50, y=30)

        btn_flag_train = tkinter.Button(self.tki, text='Train', command = self.click_flag_train)
        btn_start = tkinter.Button(self.tki, text='Start', command = self.click_start)
        label1 = tkinter.Label(self.tki,text="1. Select Train or Inference")
        label1.place(x=50, y=15)
        btn_flag_train.place(x=50, y=50)

        label2 = tkinter.Label(self.tki,text="2. Select ResNet50 or Mobilenet")
        label2.place(x=50, y=75)
        rdio_one = tkinter.Radiobutton(self.tki, text='ResNet',
                                     variable=self.radio_value_net, value=1) 
        rdio_two = tkinter.Radiobutton(self.tki, text='Mobilenet',
                                     variable=self.radio_value_net, value=2) 
        rdio_three = tkinter.Radiobutton(self.tki, text='MyNet',
                                     variable=self.radio_value_net, value=3) 
        rdio_one.place(x=50, y=95)
        rdio_two.place(x=120, y=95)
        rdio_three.place(x=200, y=95)

        label4 = tkinter.Label(self.tki,text="3. Whether spliting train-val or not")
        label4.place(x=50, y=120)
        rdio_split_one = tkinter.Radiobutton(self.tki, text='Split==True',
                                     variable=self.radio_value_split, value=1) 
        rdio_split_two = tkinter.Radiobutton(self.tki, text='Split==False',
                                     variable=self.radio_value_split, value=2) 
        rdio_split_one.place(x=50, y=140)
        rdio_split_two.place(x=160, y=140)

        label5 = tkinter.Label(self.tki,text="4. Change the proportion of train data ")
        label5.place(x=50, y=165)
        self.var_sp.set(0.7)
        spinbox = tkinter.Spinbox(self.tki, from_=0.0, to=1.0, textvariable=self.var_sp, increment=0.05)
        spinbox.place(x=50, y=185)

        label3 = tkinter.Label(self.tki,text="5. Whether augmentation or not")
        label3.place(x=50, y=210)
        rdio_split_one = tkinter.Radiobutton(self.tki, text='Augmentation==True',
                                     variable=self.radio_value_aug, value=1) 
        rdio_split_two = tkinter.Radiobutton(self.tki, text='Augmentation==False',
                                     variable=self.radio_value_aug, value=2) 
        rdio_split_one.place(x=50, y=230)
        rdio_split_two.place(x=200, y=230)

        #self.val_sp = tkinter.StringVar()
        #self.val_sp.set(0.7)
        #sp = Spinbox(self.tki, textvariable=self.val_sp, from_=0.0, to=1.0, increment=0.05)
        #sp.place(x=50, y=220)
        
        #sp.grid()

        label6 = tkinter.Label(self.tki,text="6. How many epochs the network trains ")
        label6.place(x=50, y=255)
        
        self.var_sp_epochs.set(50)
        self.sp_epochs = tkinter.Spinbox(self.tki, from_=1, to=1000, textvariable=self.var_sp_epochs, increment=5)
        self.sp_epochs.place(x=50, y=275)

        label7 = tkinter.Label(self.tki,text="Start (Close this window)")
        label7.place(x=150, y=300)
        btn_start.place(x=150, y=330)

        # Display the button window
        btn_flag_train.bind("<1>",self.tkinter_callback)
        btn_start.bind("<1>",self.tkinter_callback)
        self.tki.mainloop()

        if self.radio_value_net.get() == 1:
            self.type_backbone = "ResNet50"
            #self.layer_name_gradcam = "activation_49"
            self.img_size = self.img_sizes[0]
        elif self.radio_value_net.get() == 2 :
            self.type_backbone = "Mobilenet"
            #self.layer_name_gradcam = "conv_pw_13_relu"
            self.img_size = self.img_sizes[1]
        elif self.radio_value_net.get() == 3 :
            self.type_backbone = "MyNet"
            self.layer_name_gradcam = "conv_pw_13_relu"
            self.img_size = self.img_sizes[2]

        if self.radio_value_split.get() == 1:
            self.flag_split = True
        elif self.radio_value_split.get() == 2 :
            self.flag_split = False
        
        if self.radio_value_aug.get() == 1:
            self.flag_aug = True
        elif self.radio_value_aug.get() == 2 :
            self.flag_aug = False