# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```


------------------
To start a training, run
```
tlu7@tlu7-mbp cs-336-assignment-1 % uv run pytest -k test_main
```
and you shall see terminal output such as
```
...
step,  735 loss,  4.424985885620117
step,  740 loss,  4.25531005859375
-------------------
step,  740 validation_loss,  tensor(4.3569)
step,  745 loss,  4.4907755851745605
step,  750 loss,  4.1905741691589355
test_output,   the diamond hurt. The seal climb water and shared had no treatrawberriespite0 any trainStop fierce saw her lived happily ever puzzle 'no'. colours saying station place. She said, "Okay, Tim and all of fun. And they were playing with his animals loved to help your own” saying”ought friends.
step,  755 loss,  3.933206081390381
step,  760 loss,  4.0448994636535645
-------------------
step,  760 validation_loss,  tensor(4.3439)
step,  765 loss,  4.335960388183594
step,  770 loss,  4.419667720794678
...
```
