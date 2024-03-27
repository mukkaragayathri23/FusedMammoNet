import timm
import torch
import torch.nn as nn

def mobilenet(num_classes=5):
    mobilenet_model = timm.create_model("mobilenetv2_100", pretrained=True)
    mobilenet_num_features = mobilenet_model.classifier.in_features
    mobilenet_model.classifier = torch.nn.Linear(mobilenet_num_features, num_classes)
    return mobilenet_model
def mobilenet_tl(num_classes=5):
    # Load the best model
    model =  mobilenet(num_classes)
    save_path='/content/drive/MyDrive/fusedmammonet/models/best_model_mobilenet.pth'
    model.load_state_dict(torch.load(save_path))
    return model
    
def inceptionv3(num_classes=5):
    # InceptionV3
    inceptionv3_model = timm.create_model("inception_v3", pretrained=True)
    inceptionv3_num_features = inceptionv3_model.fc.in_features
    inceptionv3_model.fc = torch.nn.Linear(inceptionv3_num_features, num_classes)  
    return inceptionv3_model
    
def inceptionv3_tl(num_classes=5):
    # Load the best model
    model =  inceptionv3(num_classes)
    save_path='/content/drive/MyDrive/fusedmammonet/models/best_model_inceptionv3.pth'
    model.load_state_dict(torch.load(save_path))
    return model
    
def efficientnet(num_classes=5):
    # EfficientNetB0
    efficientnet_b0_model = timm.create_model("efficientnet_b0", pretrained=True)
    efficientnet_b0_num_features = efficientnet_b0_model.classifier.in_features
    efficientnet_b0_model.classifier = torch.nn.Linear(efficientnet_b0_num_features, num_classes)
    return efficientnet_b0_model
    
def efficientnet_tl(num_classes=5):
    # Load the best model
    model =  efficientnet(num_classes)
    save_path='/content/drive/MyDrive/fusedmammonet/models/best_model_efficientnet.pth'
    model.load_state_dict(torch.load(save_path))
    return model
    
def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True
    return model

def unfreeze_last_layer(model, last_layer_name='classifier'):

    if last_layer_name.lower() == 'classifier':
        for param in model.classifier.parameters():
            param.requires_grad = True

    if last_layer_name.lower() == 'fc':
        for param in model.fc.parameters():
            param.requires_grad = True

    return model
class MyEnsemble(nn.Module):

    def __init__(self, modelA, modelB, modelC, input):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC

        self.fc1 = nn.Linear(3*input, 10)
        self.fc2 = nn.Linear(10, input)

    def forward(self, x):
        out1 = self.modelA(x)
        out2 = self.modelB(x)
        out3 = self.modelC(x)

        out=torch.cat((out1, out2, out3), dim=1)

        x = self.fc1(out)
        x=self.fc2(x)

        x= torch.softmax(x, dim=1)
        return x
def ensemble(model_1,model_2,model_3,num_classes=5):
    model_1=freeze_parameters(model_1)
    model_2=freeze_parameters(model_2)
    model_3=freeze_parameters(model_3)
    model_1=unfreeze_last_layer(model_1)
    model_2=unfreeze_last_layer(model_2)
    model_3=unfreeze_last_layer(model_3,last_layer_name='fc')
    model=MyEnsemble(model_1,model_2,model_3,num_classes)
    return model
def ensemble_tl(model_1,model_2,model_3,num_classes=5):
    # Load the best model
    model =  MyEnsemble(model_1,model_2,model_3,num_classes)
    save_path='/content/drive/MyDrive/fusedmammonet/models/best_model_ensemble.pth'
    model.load_state_dict(torch.load(save_path))
    return model