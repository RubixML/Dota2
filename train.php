<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Extractors\CSV;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\PersistentModel;
use Rubix\ML\Classifiers\NaiveBayes;
use Rubix\ML\Persisters\Filesystem;

ini_set('memory_limit', '-1');

echo 'Loading data into memory ...' . PHP_EOL;

$dataset = Labeled::fromIterator(new CSV('train.csv', true));

$estimator = new PersistentModel(new NaiveBayes(), new Filesystem('dota.rbx'));

echo 'Training ...' .  PHP_EOL;

$estimator->train($dataset);

if (strtolower(readline('Save this model? (y|[n]): ')) === 'y') {
    $estimator->save();
}
