# BioGen 2025 Starter Kit (Baselines)


Please install [Anaconda](https://www.anaconda.com/distribution/) to create a conda environment as follows:
```shell script
# preparing environment
conda env create -f environment.yml
conda activate biogen2025
```


## Install Java Dependency
```shell script
wget https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.1+12/OpenJDK21U-jdk_x64_linux_hotspot_21.0.1_12.tar.gz
mkdir -p $HOME/jdk
tar -xzf OpenJDK21U-jdk_x64_linux_hotspot_21.0.1_12.tar.gz -C $HOME/jdk
export JAVA_HOME="$HOME/jdk/jdk-21.0.1+12"
export PATH="$JAVA_HOME/bin:$PATH"
```

## Install Pyserini Dependency
```shell script
conda install -c pytorch faiss-cpu -y
```


## Download and index pubmed baseline
```shell script
./build_index.sh
```
It will index 26,805,982 PubMed documents.

## Download Llama2 
Before downloading, you need to agree to Meta's license terms by visiting here: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

You may need to fill out the form to agree to the license terms. Once your request approved, run the following:


```shell script
  pip install huggingface_hub
  huggingface-cli login
  huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir llama-2-7b-chat-hf

```
Get tasks datasets from https://trec.nist.gov/act_part/act_part.html and place them in ```data``` directory.

## Run Task A Baseline
```
 cd src/
 python task_a.py
```
The submission ready file will be saved in ```data``` directory. You need to change the metadata in the ```src/task_a.py``` to add your organization and run name.

## Run Task B Baseline
```
 cd src/
 python task_b.py
```

The submission ready file will be saved in ```data``` directory. You need to change the metadata in the ```src/task_b.py``` to add your organization and run name.

