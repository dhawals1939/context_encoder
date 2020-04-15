from torch import nn

class generator(nn.Module):
    '''
    Generator
    encoder-decoder setup
    '''
    def __init__(self, img_x=128, img_y=128, channels=3):
        super().__init__() #for creating a instance of nn.module for inheritence
        self.img_x, self.img_y = img_x, img_y
        self.channels = channels

        def down_sampling(input_channels=None, output_channels=None, normalize=True):
            layers = list()

            #Conv2d Inputs
            args = {
                        'in_channels':input_channels,
                        'out_channels':output_channels,
                        'stride':2,
                        'padding':1,
                        'kernel_size':(4 ,4)
                   }

            layers.append(nn.Conv2d(**args))

            #normalize flag
            if normalize:
                layers.append(nn.BatchNorm2d(output_channels))
            
            #activation
            layers.append(nn.LeakyReLU(.2))

            return layers
        
        def up_sampling(input_channels=None, output_channels=None, normalize=True):
            layers = list()

            #Conv2dTranspose Inputs
            args = {
                        'in_channels':input_channels,
                        'out_channels': output_channels,
                        'kernel_size':(4, 4),
                        'stride':2,
                        'padding':1
                   }
            
            layers.append(nn.ConvTranspose2d(**args))

            #normalize flag
            if normalize:
                layers.append(nn.BatchNorm2d(output_channels))
            
            #activation
            layers.append(nn.ReLU())

            return layers

        def get_layers(layer_config_list, transpose=False):
            layers = list()
            
            for layer_config in layer_config_list:
                if not transpose:
                    layers += down_sampling(**layer_config)
                else:
                    layers += up_sampling(**layer_config)

            return layers
        

        def config_layer(input_channels=3, output_channels=3, normalize=True):
            return locals()


        self.encoder_layers = list()

        #encoder_layers
        self.encoder_layers.append(config_layer(self.channels, 64, False)) #-->layer 1

        self.encoder_layers.append(config_layer(64, 64, True))              #-->layer 2
                                   
        self.encoder_layers.append(config_layer(64, 128, True))             #-->layer 3
        
        self.encoder_layers.append(config_layer(128, 256, True))            #-->layer 4

        self.encoder_layers.append(config_layer(256, 512, True))            #-->layer 5

        

        self.decoder_layers = list()

        #decoder_layers
        self.decoder_layers.append(config_layer(4000, 512, True))           #-->layer 1

        self.decoder_layers.append(config_layer(512, 256, True))            #-->layer 2

        self.decoder_layers.append(config_layer(256, 128, True))            #-->layer 3
        
        self.decoder_layers.append(config_layer(128, 64, True))             #-->layer 4

        #model
        self.model = nn.Sequential(
                                        #encode Layers
                                        *get_layers(self.encoder_layers, transpose=False),
                                        
                                        #bootleneck
                                        nn.Conv2d(in_channels=512, out_channels=4000, kernel_size=(1, 1)),
                                        
                                        #decode Layers
                                        *get_layers(self.decoder_layers, transpose=True),

                                        nn.Conv2d(in_channels=64, out_channels=self.channels, kernel_size=(3, 3), stride=1, padding=1),

                                        nn.Tanh()
                                  )
    def forward(self, x):
        return self.model(x)
