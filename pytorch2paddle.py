import paddle.fluid as fluid
import torch
import collections
from model.ResNet3D import ResNet3D
def torch2paddle(torch_para, paddle_model,paddle_para_name=None):
    torch_state_dict = torch.load(torch_para)
    torch_state_dict = torch_state_dict['state_dict']

    paddel_state_dict = paddle_model.state_dict()
    #去掉bn中多余的参数
    tmp = []
    for key in torch_state_dict.keys():
        if ('_tracked' in key):
            tmp.append(key)
    for i in range(len(tmp)):
        torch_state_dict.pop(tmp[i])
    print(len(torch_state_dict))
    print(len(paddel_state_dict))
    assert(len(torch_state_dict)==len(paddel_state_dict))
    new_weight = collections.OrderedDict()
    for torch_key,paddle_key in zip(torch_state_dict.keys(),paddel_state_dict.keys()):
        tmp = torch_state_dict[torch_key].detach().numpy()
        # print(torch_key,"---",paddle_key)
        if 'fc' in torch_key  :
            # new_weight[paddle_key]=tmp.T
            # print(torch_key)
            new_weight[paddle_key] =paddel_state_dict[paddle_key]  ###finetune的时候 "fc"层不变
            
        else:
            new_weight[paddle_key] = tmp
    paddle_model.set_dict(new_weight)
    if paddle_para_name==None:
        name = torch_para[0:-4]
        fluid.save_dygraph(paddle_model.state_dict(), name)
    else:
        fluid.save_dygraph(paddle_model.state_dict(),paddle_para_name)
# # =============================================================================

if __name__ == "__main__":
    with fluid.dygraph.guard():
        paddle_model = ResNet3D(50,101)        
        torch2paddle(r'H:\0\pytorch\r3d50_KM_200ep.pth',paddle_model,r'H:\0\pytorch\resnet_3d_model')
