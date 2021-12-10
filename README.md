Development version of the Python age-household model. Run the examples from
the main repo directory. In order to make the `model` directory visible as
Python module, run the following from the main directory of the repo e.g.


On UNIX-like system
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
python examples/uk/run.py
```

On Windows, the easiest way is to install Anaconda with a graphic installer
from [here](https://www.anaconda.com/products/individual).

```
cd <directory with the code>
set PYTHONPATH=%CD%
python examples/uk/run.py
```


## Prerequisites

 * pandas - for readind and manipulating data from spreadsheets
 * tqdm - simple progress bar
