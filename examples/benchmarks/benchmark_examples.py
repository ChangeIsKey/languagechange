# Example usage of the classes in languagechange.benchmark
from languagechange.models.representation.contextualized import XL_LEXEME
from languagechange.benchmark import DWUG, SemEval2020Task1
import random
from pathlib import Path
from languagechange.usages import TargetUsage
from languagechange.benchmark import WiC, WSD, WSI

# -- WiC, WSI and WSD --

# Initialize a WiC dataset (needs dataset, language and version arguments):
wic_en = WiC(dataset='WiC', language='EN', version='1.0')

# Get all data examples:
all_data_en = wic_en.get_all_data()

# Get examples from one dataset split:
test_data = wic_en.get_test()

# Split into new partitions:
wic_en.split_train_dev_test(train_prop=0.8, dev_prop=0.19, test_prop=0.01)

# WiC datasets can also be initialized from your own target usages.
# As a list of dictionaries:
wic_examples = [
    {
        'word': 'jam',
        'text1': 'Strawberry jam',
        'text2': 'Traffic jam',
        'start1': 11, 'end1': 14,
        'start2': 8, 'end2': 11,
        'label': 0
    },
    {
        'word': 'jam',
        'text1': 'Strawberry jam',
        'text2': 'Blueberry jam',
        'start1': 11, 'end1': 14,
        'start2': 10, 'end2': 13,
        'label': 1
    }
]
custom_wic = WiC(wic_data=wic_examples)

# As a list of pairs of TargetUsage:
target_usages = [
    TargetUsage('Strawberry jam', offsets=[11, 14]),
    TargetUsage('Traffic jam', offsets=[8, 11]),
    TargetUsage('Blueberry jam', offsets=[10, 13])
]

target_usage_pairs = [
    (target_usages[0], target_usages[1]),
    (target_usages[0], target_usages[2]),
    (target_usages[1], target_usages[2])
]
# In this case, 'labels' needs to be provided as a list, where each value corresponds to the label of the pair in
# 'wic_data'.
labels = [0, 1, 0]
wic = WiC(wic_data=target_usage_pairs, labels=labels)

# Evaluate on a WiC dataset: By default, the test partition is chosen, though a different partition can be chosen with
# the ´dataset' parameter. Predictions are expected to be in the same format as the examples.
predictions = [0] * len(wic_en.get_test())  # toy predictions
f1 = wic_en.evaluate_f1(predictions)

# for graded WiC, where labels are floats, use
# wic.evaluate_spearman(predictions) # instead

# What can be done with WiC can also be done with WSD and WSI
wsd = WSD(dataset='XL-WSD', language='DA', version='no-version')
all_data = wsd.get_all_data()

# Split into new partitions
wsd.split_train_dev_test(train_prop=0.9, dev_prop=0.09, test_prop=0.01)

# WSI datasets can be initialized from custom data, or by casting from a WSD dataset. They are identical to WSD
# datasets with the only exception that evaluation is carried out differently.
wsi_examples = [
    {
        'word': 'jam',
        'text': 'Strawberry jam',
        'start': 11, 'end': 14,
        'label': 'food'
    },
    {
        'word': 'jam',
        'text': 'Blueberry jam',
        'start': 10, 'end': 13,
        'label': 'food'
    },
    {
        'word': 'jam',
        'text': 'Traffic jam',
        'start': 8, 'end': 11,
        'label': 'congestion'
    }
]

custom_wsi = WSI(wsi_examples)

# Or equivalently
target_usages = [
    TargetUsage('Strawberry jam', offsets=[11, 14], sense='food'),
    TargetUsage('Traffic jam', offsets=[8, 11]),
    TargetUsage('Blueberry jam', offsets=[10, 13])
]
labels = ['food', 'congestion', 'food']
custom_wsi = WSI(target_usages, labels=labels)

# WSD datasets can also be cast to WSI to evaluate on the WSI task.
wsi = wsd.cast_to_WSI()

# Evaluate WSD and WSI
predictions = {d['id']: d['label'] for d in wsi.get_test()}  # everything right
# predictions = {d['id']: d['word']+':0' for d in wsi.get_test()} # always predict the same
# predictions = {d['id']: d['word']+str(random.choice([0,1,2,3,4])) for d in wsi.get_test()} # predict randomly

f1 = wsd.evaluate_f1(predictions, dataset='test')
ari = wsi.evaluate_ari(predictions, dataset='test')
purity = wsi.evaluate_purity(predictions, dataset='test')

# WSD datasets can also be cast to WiC:
wsd_wic = wsd.cast_to_WiC()

# -- DWUG and SemEval --

# DWUGS are initialized with dataset ('DWUG', 'DWUG Sense' or 'NorDiaChange'), language and version parameters.
# NorDiaChange also needs an additional subset parameter, to tell which portion of the dataset to use.
dwug_sv = DWUG(dataset="DWUG", language="SV", version="3.0.0")
dwug_no = DWUG(dataset="NorDiaChange", language="NO", version="no-version", subset=1)
dwug_de_sense = DWUG(
    dataset="DWUG Sense", language="DE", version="1.0.0", config="maj_3"
)

# Get usages of a word
usages = dwug_no.get_word_usages('ørret')

# Save a graph with two supplied clusters; no model is needed for this plot.
artifact_dir = Path('benchmark_artifacts')
artifact_dir.mkdir(parents=True, exist_ok=True)
dwug_no.plot_graph(
    'leilighet', save_to_file=True,
    outfile=str(artifact_dir / 'leilighet_graph.png'), plot_cluster_labels=True,
)

# Re-annotate a word (making relatedness judgments between usages) with a computational annotator
xl_lexeme = XL_LEXEME(device="cpu")
judgments_path = artifact_dir / 'egg_judgments.csv'
clusters_path = artifact_dir / 'egg_clusters.csv'
dwug_no.annotate_word('egg', xl_lexeme, save_path=str(judgments_path))

# Cluster the word to get senses according to XL-LEXEME
dwug_no.cluster(
    'egg', judgments_file_or_dir=str(judgments_path),
    save_path=str(clusters_path), plot=False,
)

# For the whole annotation + cluster pipeline, we can also use a composed method that does the same thing
dwug_no.annotate_and_cluster(
    'egg', xl_lexeme,
    judgments_path=str(artifact_dir / 'egg_combined_judgments.csv'),
    clusters_path=str(artifact_dir / 'egg_combined_clusters.csv'), plot=False,
)

# Casting from DWUG to WiC, WSD and WSI datasets:
dwic = dwug_sv.cast_to_WiC()
dwsd = dwug_sv.cast_to_WSD()
dwsi = dwug_sv.cast_to_WSI()

all_dun_usages = dwic.get_data_by_word('all', 'dun')  # partition, word

# SemEval2020Task1 datasets are initialized with the language, as well as the subset for NO and RU
semeval_no = SemEval2020Task1('NO', subset=1)

# Like with DWUGs, we can get usages for a word
new_usages = semeval_no.get_word_usages('ørret')  # should be the same as before

# We can also evaluate on the binary or graded lexical semantic change detection task
predictions = {w: i for i, w in enumerate(semeval_no.target_words)}  # bad predictions
# predictions = {w: semeval_no.graded_change[w] for w in semeval_no.target_words}

spearman_r = semeval_no.evaluate_gcd(predictions)
