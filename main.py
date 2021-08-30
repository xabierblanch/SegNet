from SegNet import SegNet

model = SegNet()

model.summary()

keras.utils.plot_model(model, to_file='model.png', show_shapes=False,
    show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)