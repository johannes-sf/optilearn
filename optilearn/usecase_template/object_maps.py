from optilearn import object_maps


class Commons(object_maps.Commons):
    pass


class DataLoaders(object_maps.DataLoaders):
    pass


class Models(object_maps.Models):
    pass


class LossFuncs(object_maps.LossFuncs):
    pass


class UFuncs(object_maps.UFuncs):
    pass


class Envs(object_maps.Envs):
    pass


class ObjectMap(object_maps.ObjectMap):
    commons: Commons = Commons
    data_loaders: DataLoaders = DataLoaders
    models: Models = Models
    loss_funcs: LossFuncs = LossFuncs
    u_funcs: UFuncs = UFuncs
    envs: Envs = Envs
