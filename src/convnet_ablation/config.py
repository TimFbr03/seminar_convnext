experiments = {

    "baseline":
    {
        "patchify": False,
        "single_activation": False,
        "inverted_bottleneck": False,
        "layer_norm": False
    },

    "patchify":
    {
        "patchify": True,
        "single_activation": False,
        "inverted_bottleneck": False,
        "layer_norm": False
    },

    "single_activation":
    {
        "patchify": True,
        "single_activation": True,
        "inverted_bottleneck": False,
        "layer_norm": False
    },

    "inverted_bottleneck":
    {
        "patchify": True,
        "single_activation": True,
        "inverted_bottleneck": True,
        "layer_norm": False
    },

    "convnext":
    {
        "patchify": True,
        "single_activation": True,
        "inverted_bottleneck": True,
        "layer_norm": True
    }
}
