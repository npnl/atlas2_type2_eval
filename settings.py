from isles.scoring import dice_coef, volume_difference, simple_lesion_count_difference, lesion_f1_score

eval_settings = {
    "UseBIDSLoader": False,                                  # Whether to use BIDSLoader (vs. GC loader)
    "GroundTruthRoot": "/opt/evaluation/ground-truth/",     # Path to the ground truth
    "PredictionRoot": "/input/",                            # Path to the user predictions
    "GroundTruthBIDSDerivativeName": ["atlas2"],            # BIDS derivative name of the ground truth
    "PredictionBIDSDerivativeName": ["atlas2_prediction"],  # BIDS derivative name of the predictions
    "GroundTruthEntities": {                                # BIDS entities identifying the ground truth
        "subject": "",
        "session": "",
        "suffix": "mask"
    },
    "PredictionEntities": {                                 # BIDS entities identifying the predictions
        "suffix": "mask"
    },
    "GrandLoaderSettings": {"BatchSize": 2,
                            "GroundTruthPath": "/opt/evaluation/mha_masks/"},
    "LoaderBatchSize": 2,                                   # Number of images to load at a time
    "Multiprocessing": 2,                                   # Number of processors to use in parallel
    "Aggregates": ["mean", "std", "min", "max", "25%", "50%", "75%", "count", "uniq", "freq"],  # Summary stats to use
    "MetricsOutputPath": "/output/metrics.json",            # Desired location of output summary
    "SampleBIDS": "/opt/evaluation/sample_bids/",           # Path to the sample BIDS directory; don't modify.
    "ScoringFunctions": {'Dice': dice_coef,                 # Functions to use for scoring the dataset.
                         'Volume Difference': volume_difference,
                         'Simple Lesion Count': simple_lesion_count_difference,
                         'Lesionwise F1-Score': lesion_f1_score}
}


# "GrandLoaderSettings": {"InputPath": "/data1/atlas2_gc/private/hidden_mha/t1-brain-mri/",
#                         "OutputPath": "hidden_predictions/",
#                         "GroundTruthRoot": "/opt/evaluation/ground-truth/",
#                         "JSONPath": "/inputs/predictions.json",
#                         "BatchSize": 2,
#                         "InputSlugs": ["t1-brain-mri"],
#                         "OutputSlugs": ["stroke-segmentation"]}


# "GrandLoaderSettings": {"InputPath": "/data1/atlas2_gc/private/hidden_mha/t1-brain-mri/",
#                         "OutputPath": "hidden_predictions/stroke-segmentation/",
#                         "GroundTruthRoot": "hidden_mha_masks/",
#                         "JSONPath": "predictions.json",
#                         "BatchSize": 2,
#                         "InputSlugs": ["t1-brain-mri"],
#                         "OutputSlugs": ["stroke-segmentation"]},