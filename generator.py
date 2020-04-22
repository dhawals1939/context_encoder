from torch import nn

class generator(nn.Module):
    '''
    Generator
    encoder-decoder setup
    '''
    def __init__(self, img_x=128, img_y=128, channels=3):
        super().__init__()                                                  # for creating a instance of nn.module for inheritence
        self.img_x, self.img_y = img_x, img_y
        self.channels = channels

        def down_sampling(input_channels=None, output_channels=None, normalize=True, activation=True, kernel_size=None, stride=None, padding=None):
            layers = list()

            #Conv2d Inputs
            args = {
                        'in_channels':  input_channels,
                        'out_channels': output_channels,
                        'stride':       stride,
                        'padding':      padding,
                        'kernel_size':  kernel_size
                   }

            layers.append(nn.Conv2d(**args, bias=False))

            #normalize flag
            if normalize:
                layers.append(nn.BatchNorm2d(output_channels))
            
            #activation
            if activation:
                layers.append(nn.LeakyReLU(.2, True))

            return layers
        
        def up_sampling(input_channels=None, output_channels=None, normalize=True, activation=True, kernel_size=None, stride=None, padding=None):
            layers = list()

            #Conv2dTranspose Inputs
            args = {
                        'in_channels':  input_channels,
                        'out_channels': output_channels,
                        'kernel_size':  kernel_size,
                        'stride':       stride,
                        'padding':      padding
                   }
            
            layers.append(nn.ConvTranspose2d(**args, bias=False))

            #normalize flag
            if normalize:
                layers.append(nn.BatchNorm2d(output_channels))
            
            #activation
            if activation:
                layers.append(nn.ReLU(True))

            return layers

        def get_layers(layer_config_list, transpose=False):
            layers = list()
            
            for layer_config in layer_config_list:
                if not transpose:
                    layers += down_sampling(**layer_config)
                else:
                    layers += up_sampling(**layer_config)

            return layers
        

        def config_layer(input_channels=3, output_channels=3, normalize=True, activation=True, stride=2, padding=1, kernel_size=(4, 4)):
            return locals()


        self.encoder_layers = list()

        #encoder_layers
        self.encoder_layers.append(config_layer(self.channels, 64, False))                           #-->layer 1 input --> 3 x 128 x 128

        self.encoder_layers.append(config_layer(64, 64, True))                                       #-->layer 2 input --> 64 x 64 x 64
                                   
        self.encoder_layers.append(config_layer(64, 128, True))                                      #-->layer 3 input --> 64 x 32 x 32
        
        self.encoder_layers.append(config_layer(128, 256, True))                                     #-->layer 4 input --> 128 x 16 x 16

        self.encoder_layers.append(config_layer(256, 512, True))                                     #-->layer 5 input --> 256 x 8 x 8


        self.bottleneck = list()

        #bottleneck
        
        self.bottleneck += (
                                nn.Conv2d(in_channels=512, out_channels=4000, kernel_size=(4, 4), bias=False),  #-->bottleneck input --> 512 x 4 x 4
                                nn.BatchNorm2d(4000),
                                nn.LeakyReLU(.2, True),
                           )

        self.decoder_layers = list()

        #decoder_layers
        self.decoder_layers.append(config_layer(4000, 512, True, stride=1, padding=0))              #-->layer 1 input --> 4000 x 1 x 1

        self.decoder_layers.append(config_layer(512, 256, True))                                    #-->layer 2 input --> 512 x 4 x 4

        self.decoder_layers.append(config_layer(256, 128, True))                                    #-->layer 3 input --> 256 x 8 x 8
        
        self.decoder_layers.append(config_layer(128, 64, True))                                     #-->layer 4 input --> 128 x 16 x 16

        self.decoder_layers.append(config_layer(64, 3, False, False))                               #-->layer 5 input --> 64 x 32 x 32

        #model
        self.model = nn.Sequential(
                                        #encode Layers
                                        *get_layers(self.encoder_layers, transpose=False),
                                        
                                        #bootleneck
                                        *self.bottleneck,
                                        
                                        #decode Layers
                                        *get_layers(self.decoder_layers, transpose=True),

                                        nn.Tanh()
                                        #output --> 3 x 64 x 64
                                  )
    def forward(self, x):
        return self.model(x)
