# Rubix ML - Dota 2 Game Outcome Predictor

[Dota 2](http://www.dota2.com/) is a popular multiplayer online battle arena (MOBA) game that puts 10 players divided into 2 teams against each other. Each player controls a unique hero with abilities and its own set of strengths and weaknesses. Our objective is to build a classifier to predict the winning team based on hero matchup given a dataset of 102,944 individual matchups and their labeled outcomes. We'll employ the [Naive Bayes](https://rubixml.github.io/ML//latest/classifiers/naive-bayes.html) algorithm as our base estimator and learn how to save the trained model for use in another process. We'll also test the model to see how well it can generalize what it has learned to new data.

## Installation

Clone the project locally using [Composer](https://getcomposer.org/):

```sh
$ composer create-project rubix/dota2
```

## Requirements

- [PHP](https://php.net) 8.3 or above

## Tutorial

Radiant versus Dire. Ten players, one Ancient to guard, and a draft that tilts or soars before the first creep wave even spawns. Every Dota player has been there: staring at the scoreboard during the pick phase, muttering *"no way our lineup wins this."* This project is that intuition, minus the resignation—we're going to teach a machine how to judge a draft, using the outcome of over 100,000 real matchups.

Our crystal ball of choice is [Naive Bayes](https://rubixml.github.io/ML/latest/classifiers/naive-bayes.html), a classifier that counts how often each hero shows up on a winning team and uses those counts to guess whether a lineup is Radiant-shenanigans or Dire-trouble. Along the way we'll also learn how to freeze a trained model to disk and cook up a fresh batch of matchups to test just how cocky the model gets.

### Understand the Data

Each row in the dataset is one completed game. The `hero_1` through `hero_113` columns encode the draft: a `1` means the hero fought for `team_1`, a `-1` means it fought for `team_2`, and a `0` means it sat the game out (probably loading a game of Azeroth). The final column, `outcome`, tells us which team took home the W:

```
cluster_id,game_mode,game_type,hero_1,...,hero_113,outcome
...
223,2,2,0,0,0,0,0,0,0,0,0,1,...,0,team_2
```

### Train the Model

[`train.php`](train.php) is where the model gets its education. It slurps the training set into memory, fits a Naive Bayes classifier to it, and—if you say the magic word—saves the trained model to disk so we can use it again later.

```php
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
```

First, the [CSV](https://rubixml.github.io/ML/latest/extractors/csv.html) extractor rips through `train.csv` line by line—no need to fit the whole 92,649 matchups in memory at once—and `Labeled::fromIterator()` turns the stream into a labeled dataset, using the header row to figure out which column (`outcome`) is the response variable. No label, no learning. Just like queueing without a role.

Next comes the star of the show:

```php
new PersistentModel(new NaiveBayes(), new Filesystem('dota.rbx'));
```

The `PersistentModel` wrapper wraps our `NaiveBayes` estimator in a `Filesystem` persister, which means we can save the trained model to a file on disk and load it back up later—no retraining, no grinding, no 5-stack required. Naive Bayes learns by counting how often each hero appears on each team and how often each team wins, then wielding those counts to predict new matchups. It makes a very bold assumption that every hero contributes independently to the outcome, i.e. that heroes never coordinate with each other. We all know that's a lie after someone first-picks Pudge and feeds, but it works surprisingly well anyway.

The `train()` call is where the magic happens. Sit back, admire those victory probabilities, and when the script asks *"Save this model?"* hit `y` to drop the trained `dota.rbx` file. Go ahead and run it:

```sh
composer train
```

### Validate the Model

Every good pub has an enemy, and every good model has a test set. We held out over 10,000 matchups (`test.csv`) to see if our model generalizes, or whether it memorized the draft order and crumbles the moment someone surprises with a mid Techies.

[`validate.php`](validate.php) loads the saved model back from `dota.rbx`, predicts the outcome of every matchup in the test set, and grades the model with a report card worthy of a battle pass:

```php
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
```

The model gets resurrected from the grave with `PersistentModel::load()`—the same persister we used to save it, proving both sides of the file cabinet work. Then `predict()` lets it call the shots on matchups it has never seen. Naive Bayes doesn't so much *predict* as *compute*, but it sounds cooler with a verb that implies prophecy.

Finally, the moment of truth: the [MulticlassBreakdown](https://rubixml.github.io/ML/latest/cross-validation/reports/multiclass-breakdown.html) reports accuracy, precision, and recall for each class, while the [ConfusionMatrix](https://rubixml.github.io/ML/latest/cross-validation/reports/confusion-matrix.html) shows exactly which lineups had the model second-guessing itself. Both reports are bundled into an `AggregateReport`, printed to the console, and saved as `report.json` for your competitive-nerd spreadsheets.

See how your oracle performs:

```sh
composer test
```

And there you have it—a model that never whiffs a last-hits check, survives the ground without a courier, and calls the winner before the horn sounds. Its MMR is almost certainly higher than your account's. Report back with your accuracy and try to convince yourself it isn't just confirming your own draft PTSD.

## Original Dataset

stephen.tridgell '@' sydney.edu.au

## References

>- Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

## License

The code is licensed [MIT](LICENSE) and the tutorial is licensed [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
