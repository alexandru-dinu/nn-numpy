class Network:
    def __init__(self, layers):
        self.layers = layers
        self.num_layers = len(self.layers)

        
    def forward(self, x):
        prev_x = x
        
        for layer in self.layers:
            prev_x = layer.forward(prev_x)
            
        return prev_x

    
    def backward(self, x, out):
        crt_err = out

        for l in range(self.num_layers - 1, 0, -1):
            crt_layer = self.layers[l]
            prev_layer = self.layers[l - 1]

            crt_err = crt_layer.backward(prev_layer.outputs, crt_err)

        self.layers[0].backward(x, crt_err)

        
    def update_parameters(self, alpha):
        for layer in self.layers:
            layer.update_parameters(alpha)

            
    def __str__(self):
        return " -> ".join(map(str, self.layers))
    
    
    def __repr__(self):
        return str(self)