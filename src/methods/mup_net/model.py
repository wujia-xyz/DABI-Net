import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models.vgg

#################################################################
# VGG
#################################################################
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'end'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG_features(nn.Module):
    def __init__(self, cfg, batch_norm=False, init_weights=True):
        super(VGG_features, self).__init__()

        self.batch_norm = batch_norm

        self.kernel_sizes = []
        self.strides = []
        self.paddings = []

        self.features = self._make_layers(cfg, batch_norm)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, batch_norm):

        self.n_layers = 0

        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

                self.kernel_sizes.append(2)
                self.strides.append(2)
                self.paddings.append(0)

            elif v != "end":
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]

                self.n_layers += 1

                self.kernel_sizes.append(3)
                self.strides.append(1)
                self.paddings.append(1)

                in_channels = v

        return nn.Sequential(*layers)

    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings

    def num_layers(self):
        '''
        the number of conv layers in the network
        '''
        return self.n_layers

    # def __repr__(self):
    #     template = 'VGG{}, batch_norm={}'
    #     return template.format(self.num_layers() + 3,
    #                            self.batch_norm)

def vgg16_features(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_features(cfg['D'], batch_norm=False, **kwargs)
    return model

# if __name__ == '__main__':
#     fts = vgg16_features(pretrained=True)
#     inp = torch.rand([2,3,128,128])
#     out = fts(inp)
#     print(out.shape) # [2, 512, 8, 8]

#################################################################
# ResNet
#################################################################
import torchvision.models as models

class ResNet_features(nn.Module):
    def __init__(self, pretrained):
        super(ResNet_features, self).__init__()
        # wegihts=models.ResNet18_Weights.IMAGENET1K_V1
        model_ft1 = models.resnet18(pretrained=pretrained)
        self.conv1 = model_ft1.conv1  # kernel_size=7, stride=2, padding=3
        self.bn1 = model_ft1.bn1
        # [2, 2, 2, 2] * BasicBlock
        self.layer1 = model_ft1.layer1 # [(k=3, s=1, p=1),(k=3, s=1, p=1)] * 2
        self.layer2 = model_ft1.layer2 # [(k=3, s=2, p=1),(k=3, s=1, p=1)] + [(k=3, s=1, p=1),(k=3, s=1, p=1)]
        self.layer3 = model_ft1.layer3 # [(k=3, s=2, p=1),(k=3, s=1, p=1)] + [(k=3, s=1, p=1),(k=3, s=1, p=1)]
        self.layer4 = model_ft1.layer4 # [(k=3, s=2, p=1),(k=3, s=1, p=1)] + [(k=3, s=1, p=1),(k=3, s=1, p=1)]
        self.relu = nn.ReLU()

        self.kernel_sizes = []
        self.strides = []
        self.paddings = []
    def forward(self, x):
        x_1 = self.conv1(x) # 1/2
        x_1 = self.bn1(x_1)
        x_1 = self.relu(x_1)
        x_1 = self.layer1(x_1) # 1/2
        x_1 = self.layer2(x_1) # 1/4
        x_1 = self.layer3(x_1) # 1/8
        x_1 = self.layer4(x_1) # 1/16
        return x_1

    def conv_info(self):
        self.kernel_sizes = [7, 3,3,3,3, 3,3,3,3, 3,3,3,3, 3,3,3,3]
        self.strides      = [2, 1,1,1,1, 2,1,1,1, 2,1,1,1, 2,1,1,1]
        self.paddings     = [3, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1]
        return self.kernel_sizes, self.strides, self.paddings


def resnet18_features(pretrained=False):
    return ResNet_features(pretrained)

# if __name__ == '__main__':
#     fts = resnet18_features(pretrained=True)
#     inp = torch.rand([2,3,128,128])
#     out = fts(inp)
#     print(out.shape) # [2, 512, 8, 8]

#################################################################
# PPNet
#################################################################
class PPNet(nn.Module):
    def __init__(self, features, img_size, prototype_shape,
                 topk_k=1, num_classes=3, init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck',
                 class_specific=False):

        super(PPNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.topk_k = topk_k # for a 14x14: topk_k=3 is 1.5%, topk_k=9 is 4.5%
        self.num_classes = num_classes
        self.class_specific=class_specific
        self.epsilon = 1e-4

        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function

        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        '''
        assert(self.num_prototypes % self.num_classes == 0)

        # this has to be named features to allow the precise loading
        self.features = features

        features_name = str(self.features).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')

        if add_on_layers_type == 'regular':
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
            )
        else:
            raise NotImplementedError()

        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)

        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # forward
    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        l2 distance
        '''
        # e.g. ones -> prototype_shape=(num_prototypes, 512, 1, 1)
        #      x -> [b, 512, h, w]
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones) # [b, num_prototypes, h, w]

        p2 = self.prototype_vectors ** 2 # (num_prototypes, 512, 1, 1)
        p2 = torch.sum(p2, dim=(1, 2, 3)) # (num_prototypes,)
        p2_reshape = p2.view(-1, 1, 1) # (num_prototypes, 1, 1)

        xp = F.conv2d(input=x, weight=self.prototype_vectors) # [b, num_prototypes, h, w]
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances # [b, num_prototypes, h, w]
    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.features(x)
        x = self.add_on_layers(x)
        return x
    def prototype_distances(self, x):
        '''
        x is the raw input
        '''
        conv_features = self.conv_features(x)
        distances = self._l2_convolution(conv_features)
        return distances

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def forward(self, x):
        # x is of dimension (batch, 4, spatial, spatial)
        x = x[:, 0:3, :, :]
        distances = self.prototype_distances(x)
        '''
        we cannot refactor the lines below for similarity scores
        because we need to return min_distances
        '''
        # global min pooling
        _distances = distances.view(distances.shape[0], distances.shape[1], -1)
        top_k_neg_distances, _ = torch.topk(-_distances, self.topk_k)
        closest_k_distances = - top_k_neg_distances
        min_distances = F.avg_pool1d(closest_k_distances, kernel_size=closest_k_distances.shape[2])
        min_distances = min_distances.view(-1, self.num_prototypes)
        #
        prototype_activations = self.distance_2_similarity(distances)
        _activations = prototype_activations.view(prototype_activations.shape[0], prototype_activations.shape[1], -1)
        top_k_activations, _ = torch.topk(_activations, self.topk_k)
        prototype_activations = F.avg_pool1d(top_k_activations, kernel_size=top_k_activations.shape[2])
        prototype_activations = prototype_activations.view(-1, self.num_prototypes) # [b, nprotos]

        logits = prototype_activations # !!! NO last_layer anymore

        if not self.class_specific:
            logits[:,0] = 0

        activation = self.distance_2_similarity(distances)
        upsampled_activation = torch.nn.Upsample(size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)(activation)

        return logits, min_distances, upsampled_activation

    def push_forward(self, x):
        '''this method is needed for the pushing operation'''
        conv_output = self.conv_features(x)
        distances = self._l2_convolution(conv_output)
        return conv_output, distances


#################################################################
# PPNet3
#################################################################
class PPNet3(nn.Module):
    def __init__(self, features, img_size, prototype_shape,
                 topk_k=1, num_classes=3, init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck',
                 class_specific=False):
        super(PPNet3, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.prototype_shape_l = [prototype_shape] * 3
        self.num_prototypes_l = [prototype_shape[0]] * 3
        self.pnet123 =nn.ModuleList([PPNet(features=features[0],
                            img_size=img_size,
                            prototype_shape=prototype_shape,
                            topk_k=topk_k,
                            num_classes=num_classes,
                            init_weights=True,
                            prototype_activation_function=prototype_activation_function,
                            add_on_layers_type=add_on_layers_type,
                            class_specific=class_specific),
                        PPNet(features=features[1],
                            img_size=img_size,
                            prototype_shape=prototype_shape,
                            topk_k=topk_k,
                            num_classes=num_classes,
                            init_weights=True,
                            prototype_activation_function=prototype_activation_function,
                            add_on_layers_type=add_on_layers_type,
                            class_specific=class_specific),
                        PPNet(features=features[2],
                            img_size=img_size,
                            prototype_shape=prototype_shape,
                            topk_k=topk_k,
                            num_classes=num_classes,
                            init_weights=True,
                            prototype_activation_function=prototype_activation_function,
                            add_on_layers_type=add_on_layers_type,
                            class_specific=class_specific)])

        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(sum(self.num_prototypes_l),
                                                    self.num_classes)
        if not class_specific:
            raise NotImplementedError
        else:
            offset = 0
            for i in range(3):
                num_prototypes_per_class = self.num_prototypes_l[i] // self.num_classes
                if i == 0:
                    val = 1
                elif i == 1:
                    val = 1
                elif i == 2:
                    val = 1
                for j in range(self.num_prototypes_l[i]):
                    self.prototype_class_identity[offset+j, j // num_prototypes_per_class] = val
                offset += self.num_prototypes_l[i]

        self.last_layer = nn.Linear(sum(self.num_prototypes_l), self.num_classes,
                                    bias=False) # do not use bias
        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

    # initial weights
    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights = torch.t(self.prototype_class_identity)
        positive_one_weights_locations = torch.t(self.prototype_class_identity) > 0
        negative_one_weights_locations = 1 - positive_one_weights_locations.float()

        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            positive_one_weights + incorrect_class_connection * negative_one_weights_locations)

    def prune_prototypes(self, prototypes_to_prune_l):
        ''' prune prototypes for three modalities simultaneously
        prototypes_to_prune_l: a list of indices list each in
        [0, current number of prototypes - 1] that indicates the prototypes to
        be removed
        '''
        prototypes_to_keep = []
        offset = 0
        for mi,prototypes_to_prune in enumerate(prototypes_to_prune_l):
            old_num_protos = self.num_prototypes_l[mi]
            to_keep = list(set(range(old_num_protos)) - set(prototypes_to_prune))
            pnet = self.pnet123[mi]
            # protos vector
            pnet.prototype_vectors = nn.Parameter(pnet.prototype_vectors.data[to_keep, ...],
                                                  requires_grad=True)
            pnet.prototype_shape = list(pnet.prototype_vectors.size())
            self.prototype_shape_l[mi] = pnet.prototype_shape
            pnet.num_prototypes = pnet.prototype_shape[0]
            self.num_prototypes_l[mi] = pnet.num_prototypes
            # self.ones is nn.Parameter
            pnet.ones = nn.Parameter(pnet.ones.data[to_keep, ...], requires_grad=False)
            # combine
            prototypes_to_keep.extend([e+offset for e in to_keep])
            offset += old_num_protos
        # self.prototype_class_identity is torch tensor
        # so it does not need .data access for value update
        self.prototype_class_identity = self.prototype_class_identity[prototypes_to_keep, :]
        # changing self.last_layer in place
        # changing in_features and out_features make sure the numbers are consistent
        self.last_layer.in_features = sum(self.num_prototypes_l)
        self.last_layer.out_features = self.num_classes
        self.last_layer.weight.data = self.last_layer.weight.data[:, prototypes_to_keep]

    def forward(self, x):
        x1 = x[:,0,:,:,:]
        x2 = x[:,1,:,:,:]
        x3 = x[:,2,:,:,:]
        logits1, min_distances1, upsampled_activation1 = self.pnet123[0](x1)
        logits2, min_distances2, upsampled_activation2 = self.pnet123[1](x2)
        logits3, min_distances3, upsampled_activation3 = self.pnet123[2](x3)
        logits = self.last_layer(torch.cat([F.relu(logits1), F.relu(logits2), F.relu(logits3)], 1))

        return logits, [min_distances1, min_distances2, min_distances3], None

class Net3(nn.Module):
    def __init__(self, features, num_classes=3):
        super(Net3, self).__init__()
        self.num_classes = num_classes
        self.net123 = nn.ModuleList(features)

        first_add_on_layer_in_channels = \
            [i for i in features[0].modules() if isinstance(i, nn.Conv2d)][-1].out_channels

        self.add_on_layers = nn.Sequential(
            nn.Conv2d(in_channels=first_add_on_layer_in_channels*3, \
                out_channels=first_add_on_layer_in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=first_add_on_layer_in_channels, \
                out_channels=first_add_on_layer_in_channels // 2, kernel_size=1),
            nn.ReLU()
        )
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.last_layer = nn.Linear(first_add_on_layer_in_channels // 2, \
                                    self.num_classes, bias=False) # do not use bias

    def forward(self, x):
        x1 = x[:,0,:,:,:]
        x2 = x[:,1,:,:,:]
        x3 = x[:,2,:,:,:]
        feats1 = self.net123[0](x1)
        feats2 = self.net123[1](x2)
        feats3 = self.net123[2](x3)
        fuse = self.add_on_layers(torch.cat([F.relu(feats1), F.relu(feats2), F.relu(feats3)], 1))
        fuse = self.pooling(fuse).squeeze(-1).squeeze(-1)
        logits = self.last_layer(fuse)
        return logits
