<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Extractors\CSV;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;

ini_set('memory_limit', '-1');

echo 'Loading data into memory ...' . PHP_EOL;

$dataset = Labeled::fromIterator(new CSV('test.csv', true));

$estimator = PersistentModel::load(new Filesystem('dota.rbx'));

echo 'Making predictions ...' . PHP_EOL;

$predictions = $estimator->predict($dataset);

$report = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);

$results = $report->generate($predictions, $dataset->labels());

echo $results;

$results->toJSON()->saveTo(new Filesystem('report.json'));

echo 'Report saved to report.json' . PHP_EOL;
