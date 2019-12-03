#可以根据自己的情况进行修改
class MyConfigs():

    data_folder = './data/flowers/'
    test_data_folder = ""
    model_name = "resnet" #Vgg ResNet152 myModel
    weights = "./checkpoints/"
    logs = "./logs/"
    example_folder = "./example/"
    freeze = True
    #
    epochs = 300
    batch_size = 16
    img_height = 227  #网络输入的高和宽
    img_width = 227
    num_classes = 20
    lr = 1e-2
    lr_decay = 1e-4
    weight_decay = 2e-4
    ratio = 0.2
config = MyConfigs()
