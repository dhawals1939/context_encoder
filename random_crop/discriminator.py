from torch import nn

class discriminator(nn.Module):
    '''
    Discrimator setup
    '''
    def __init__(self, channels=3):
        super().__init__()
        self.channels = channels

        def down_sampling(input_channels=None, output_channels=None, stride=None, kernel_size=None, normalize=None, activation=True):
            layers = list()

            #Conv2d Inputs
            args = {
                        'in_channels':input_channels,
                        'out_channels':output_channels,
                        'stride':stride,
                        'kernel_size':kernel_size,
                        'padding':1
                   }

            layers.append(nn.Conv2d(**args))

            #normalize flag
            if normalize:
                layers.append(nn.InstanceNorm2d(output_channels))
            if activation:
                layers.append(nn.LeakyReLU(.2, inplace=True))

            return layers
        
        def get_layers(layer_config_list):
            layers = list()
            
            for layer_config in layer_config_list:
                layers += down_sampling(**layer_config)
            
            return layers

        def config_layer(input_channels=3, output_channels=3, stride=2, normalize=True, kernel_size=(3, 3), activation=True):
            return locals()

        self.discriminator_layers = list()

        # Discriminator layers
        self.discriminator_layers.append(config_layer(self.channels, 64, 2, False))                                         #layer-->1

        self.discriminator_layers.append(config_layer(64, 128, 2, True))                                                    #layer-->2

        self.discriminator_layers.append(config_layer(128, 256, 2, True))                                                   #layer-->3

        self.discriminator_layers.append(config_layer(256, 512, 1, True))                                                   #layer-->4

        self.discriminator_layers.append(config_layer(512, 1, 1, False, activation=False))                                  #layer-->5

        # Model
        self.model = nn.Sequential(
                                        *get_layers(self.discriminator_layers),
                                  )
    def forward(self, x):
        output = self.model(x)
        return output