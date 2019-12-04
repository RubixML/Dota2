<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\PersistentModel;
use Rubix\ML\Classifiers\NaiveBayes;
use Rubix\ML\Persisters\Filesystem;
use League\Csv\Reader;

ini_set('memory_limit', '-1');

echo 'Loading data into memory ...' . PHP_EOL;

$reader = Reader::createFromPath(__DIR__ . '/train.csv')->setHeaderOffset(0);

$samples = $reader->getRecords([
    'cluster_id', 'game_mode', 'game_type', 'hero_1', 'hero_2', 'hero_3',
    'hero_4', 'hero_5', 'hero_6', 'hero_7', 'hero_8', 'hero_9', 'hero_10',
    'hero_11', 'hero_12', 'hero_13', 'hero_14', 'hero_15', 'hero_16', 'hero_17',
    'hero_18', 'hero_19', 'hero_20', 'hero_21', 'hero_22', 'hero_23', 'hero_24',
    'hero_25', 'hero_26', 'hero_27', 'hero_28', 'hero_29', 'hero_30', 'hero_31',
    'hero_32', 'hero_33', 'hero_34', 'hero_35', 'hero_36', 'hero_37', 'hero_38',
    'hero_39', 'hero_40', 'hero_41', 'hero_42', 'hero_43', 'hero_44', 'hero_45',
    'hero_46', 'hero_47', 'hero_48', 'hero_49', 'hero_50', 'hero_51', 'hero_52',
    'hero_53', 'hero_54', 'hero_55', 'hero_56', 'hero_57', 'hero_58', 'hero_59',
    'hero_60', 'hero_61', 'hero_62', 'hero_63', 'hero_64', 'hero_65', 'hero_66',
    'hero_67', 'hero_68', 'hero_69', 'hero_70', 'hero_71', 'hero_72', 'hero_73',
    'hero_74', 'hero_75', 'hero_76', 'hero_77', 'hero_78', 'hero_79', 'hero_80',
    'hero_81', 'hero_82', 'hero_83', 'hero_84', 'hero_85', 'hero_86', 'hero_87',
    'hero_88', 'hero_89', 'hero_90', 'hero_91', 'hero_92', 'hero_93', 'hero_94',
    'hero_95', 'hero_96', 'hero_97', 'hero_98', 'hero_99', 'hero_100', 'hero_101',
    'hero_102', 'hero_103', 'hero_104', 'hero_105', 'hero_106', 'hero_107',
    'hero_108', 'hero_109', 'hero_110', 'hero_111', 'hero_112', 'hero_113',
]);

$labels = $reader->fetchColumn('outcome');

$dataset = Labeled::fromIterator($samples, $labels);

$estimator = new PersistentModel(new NaiveBayes(2.0), new Filesystem('dota.model'));

echo 'Training ...' .  PHP_EOL;

$estimator->train($dataset);

if (strtolower(readline('Save this model? (y|[n]): ')) === 'y') {
    $estimator->save();
}