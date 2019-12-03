import torch
import cv2
import torch
from torch.utils.data import DataLoader
from torch import nn ,optim
from torch.autograd import Variable
from config import config
from datasets import *
import Model
from utils.utils import accuracy
classes= {0:"roses",1:"tulips",2:"dandelion",3:"sunflowers"}

#用于评估模型
def evaluate(test_loader,model,criterion):
    sum = 0
    test_loss_sum = 0
    test_top1_sum = 0
    model.eval()

    for ims, label in test_loader:
        input_test = Variable(ims).cuda()
        target_test = Variable(torch.from_numpy(np.array(label)).long()).cuda()
        output_test = model(input_test)
        loss = criterion(output_test, target_test)
        top1_test = accuracy(output_test, target_test, topk=(1,))
        sum += 1
        test_loss_sum += loss.data.cpu().numpy()
        test_top1_sum += top1_test[0].cpu().numpy()[0]
    avg_loss = test_loss_sum / sum
    avg_top1 = test_top1_sum / sum
    return avg_loss, avg_top1


def test(test_loader,model):
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    predict_file = open("%s.txt" % config.model_name, 'w')
    for i, (input,filename) in enumerate(tqdm(test_loader)):
        if torch.cuda.is_available():
            input = Variable(input).cuda()
        else:
            input= Variable(input)
        #print("input.size = ",input.data.shape)
        y_pred = model(input)
        smax = nn.Softmax(1)
        smax_out = smax(y_pred)
        pred_label = np.argmax(smax_out.cpu().data.numpy())
        predict_file.write(filename[0]+', ' +classes[pred_label]+'\n')
        #print(filename[0],"的预测的结果为：",labelText[pred_label])


def test_one_image(image,model):


    model.eval()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (config.img_height, config.img_width))
    img = transforms.ToTensor()(image)
    img = img.unsqueeze(0)  # 增加一个维度

    img = Variable(img)


    y_pred = model(img)
    smax = nn.Softmax(1)
    smax_out = smax(y_pred)
    pred_label = np.argmax(smax_out.cpu().data.numpy())
    # print(smax_out.cpu().data.numpy())
    # print(pred_label)
    # print(smax_out.cpu().data.numpy()[0][pred_label])
    if pred_label == 0:
        result = '这是玫瑰花的概率为：%.4f'%smax_out.cpu().data.numpy()[0][pred_label]

    elif pred_label == 1:
        result = '这是郁金香的概率为：%.4f' % smax_out.cpu().data.numpy()[0][pred_label]
    elif pred_label ==2:
        result = '这是蒲公英的概率为：%.4f' % smax_out.cpu().data.numpy()[0][pred_label]
    elif pred_label == 3:
        result = '这是向日葵的概率为：%.4f' % smax_out.cpu().data.numpy()[0][pred_label]

    return result

if __name__ == '__main__':

    #1. 定义测试集
    test_list, _ = get_files(config.data_folder,config.ratio)
    test_loader = DataLoader(datasets(test_list, transform=None,test = True), batch_size= 1, shuffle=False,
                             collate_fn=collate_fn, num_workers=4)   # 测试时这里的batch_size = 1

    #2. 加载模型及其参数
    model = Model.get_net()
    checkpoint = torch.load(config.weights+ config.model_name+'.pth')
    model.load_state_dict(checkpoint["state_dict"])
    #optimizer.load_state_dict(checkpoint["optimizer"])
    print("Start Test.......")
    test(test_loader,model)

