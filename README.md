# unmuteai
An application for performing sentiment analysis and sentiment-based recommendation engine.

### unmuteai-cli usage
#### loading data.
```python unmuteai-cli.py --run init```

#### Data preprocess
```python unmuteai-cli.py --run preprocess```

#### Split data to train set and test set.
```python unmuteai-cli.py --run split-data```

#### Baseline accuracy
```python unmuteai-cli.py --run benchmark```

#### Train model with train set
```python unmuteai-cli.py --run train --model naive_bayes```
```python unmuteai-cli.py --run train --model rf```

#### Test model with test set
```python unmuteai-cli.py --run test --model naive_bayes```
```python unmuteai-cli.py --run test --model rf```

#### Predicting with a particular model
```python unmuteai-cli.py --run predict --model naive_bayes --predict "The game is amazing" ```

```python unmuteai-cli.py --run predict --model rf --predict "The game is terrible" ```


#### Predicting with game title... 
```python unmuteai-cli.py --run predict --model naive_bayes --title --predict "The best game ever played" --title "The Guest"```
