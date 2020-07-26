def get_all_layers(model):
    """ Get all layers of model, including ones inside a nested model """
    layers = []
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            layers += get_all_layers(layer)
        else:
            layers.append(layer)
    return layers
