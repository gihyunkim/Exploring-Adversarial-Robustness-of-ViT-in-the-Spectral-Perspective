from torchvision import models
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torchvision.transforms as transforms

class GetModel:
    @staticmethod
    def getModel(model_name, pretrained=True):
        if model_name == "resnet50":
            model = models.resnet50(pretrained=pretrained)
        elif model_name == "resnet152":
            model = models.resnet152(pretrained=pretrained)
        elif model_name == "vit-b-1k":
            model = models.vit_b_16(pretrained=pretrained)
        elif model_name == "vit-b-21k":
            model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
            config = resolve_data_config({}, model=model)
            print(create_transform(**config))
            # 248
            transform = transforms.Compose([
                transforms.Resize(size=224, interpolation=transforms.InterpolationMode("bicubic")),
                transforms.CenterCrop(size=(224, 224)),
                transforms.ToTensor(),
            ])
        elif model_name == "vit-l":
            model = timm.create_model('vit_large_patch16_224', pretrained=pretrained)
            config = resolve_data_config({}, model=model)
            print(create_transform(**config))
            transform = transforms.Compose([
                transforms.Resize(size=248, interpolation=transforms.InterpolationMode("bicubic")),
                transforms.CenterCrop(size=(224, 224)),
                transforms.ToTensor(),
            ])
        elif model_name == "swin-b":
            model = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained)
            config = resolve_data_config({}, model=model)
            print(create_transform(**config))
            #248
            transform = transforms.Compose([
                transforms.Resize(size=224, interpolation=transforms.InterpolationMode("bicubic")),
                transforms.CenterCrop(size=(224, 224)),
                transforms.ToTensor(),
            ])
        elif model_name == "deit-s":
            model = timm.create_model("deit_small_distilled_patch16_224", pretrained=pretrained)
            config = resolve_data_config({}, model=model)
            print(create_transform(**config))
            #248
            transform = transforms.Compose([
                transforms.Resize(size=224, interpolation=transforms.InterpolationMode("bicubic")),
                transforms.CenterCrop(size=(224, 224)),
                transforms.ToTensor(),
            ])
        elif model_name == "deit-s-nodist":
            model = timm.create_model("deit_small_patch16_224", pretrained=pretrained)
            config = resolve_data_config({}, model=model)
            print(create_transform(**config))
            transform = transforms.Compose([
                transforms.Resize(size=248, interpolation=transforms.InterpolationMode("bicubic")),
                transforms.CenterCrop(size=(224, 224)),
                transforms.ToTensor(),
            ])
        else:
            print("not correct model name...., use model in [resnet, vgg, inception, mobilenet")
            exit(-1)

        '''cnn transforms'''
        if model_name in ["resnet50", "resnet152", "vit-b-1k","swin-b", "deit-s", "deit-s-nodist"]:
            if not model_name in ["swin-b" ,"deit-s", "deit-s-nodist"]:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((224, 224)),
                ])
            normalize = transforms.Compose([
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225))
                ])

            invNormalize = transforms.Compose([
                transforms.Normalize(mean=(0., 0., 0.),
                                     std=(1/0.229, 1/0.224, 1/0.225)),
                transforms.Normalize(mean=(-0.485, -0.456, -0.406),
                                     std=(1.,1.,1.))
            ])
        else:
            normalize = transforms.Compose([
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5))
            ])
            invNormalize = transforms.Compose([
                transforms.Normalize(mean=(0., 0., 0.),
                                     std=(1 / 0.5, 1 / 0.5, 1 / 0.5)),
                transforms.Normalize(mean=(-0.5, -0.5, -0.5),
                                     std=(1., 1., 1.))
            ])

        return model, transform, normalize, invNormalize

if __name__ == "__main__":
    get_model = GetModel()
    model_name = "resnet50"
    input_shape = (224, 224)
    model, _, _, _ = get_model.getModel(model_name, pretrained=False)
    # summary(model, (3, input_shape[0], input_shape[1]))
    # macs, params = get_model_complexity_info(model, (3, input_shape[0], input_shape[1]), as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
