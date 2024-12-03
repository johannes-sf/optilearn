from optilearn.meta_constructor import MetaConstructor


# ToDo: expand with all available constructor choices
def test_meta_constructor_init_from_config():
    conf = {
        "model_type": "nn",
        "env_name": "classification",
        "loss_type": "cross_entropy",
        "u_func_name": "linear",
        "training_data_loader": "simple",
        "eval_data_loader": "simple",
        "training_transformation_pipeline": "default",
        "eval_transformation_pipeline": "default",
    }

    constructor = MetaConstructor.from_config(conf)
