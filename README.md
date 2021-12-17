# Text-Clustering
This project groups the news in the provided news dataset to various clusters using unsupervised learning.

## Installation Instructions
The steps here are required only if you want to view/run the Jupyter Notebook (i.e. .ipynb and .py files) on your local machine. If not, you can just proceed to ***Viewing Instructions: To view the HTML version of the clustering results***, which does not require any prior setup.
#### To run the scripts on your local machine (Assumes that you already have an existing anaconda (Windows/Mac) or conda installation (Linux), with Python installed):
1. Launch Anaconda Prompt (Windows/Mac) / Terminal (Linux).
2. Navigate to Text-Clustering folder (Command below assumes that you downloaded and extracted the Text-Clustering folder to your downloads folder).
```
$ cd downloads 
$ cd Text-Clustering
```
3. Create a new Python environment and activate it.
```
$ conda create -n text-clustering python=3.7
$ conda activate text-clustering
```
4. Install the required dependencies
```
$ pip install -r requirements.txt
```
5. Launch Jupyter Notebook
```
$ jupyter notebook
```
## Viewing Instructions
#### To view the Jupyter Notebook version of the clustering results:
- From the Home Page of the Jupyter Notebook, navigate to and open Text-Clustering-with-Top2Vec.ipynb.
- Run all the cells in the notebook from top to bottom (Need to execute this step to view the interactive visualizations).
#### To view the HTML version of the clustering results:
- Open Text-Clustering-with-Top2Vec.html in your browser.
#### To view the helper scripts used in Text-Clustering-with-Top2Vec.ipynb:
- Navigate to and open top2vec.py.

## Files
Text-Clustering-with-Top2Vec.ipynb
> - Jupyter Notebook containing the clustering results
> 
Text-Clustering-with-Top2Vec.html
> - HTML version of Text-Clustering-with-Top2Vec.ipynb
> 
top2vec.py
> - Contains helper classes and functions for clustering the news dataset
> 
requirements.txt
> - Contains the list of dependencies required to run Text-Clustering-with-Top2Vec.ipynb and top2vec.py
> 
README.md
> - Contains the instructions for viewing the clustering results
>
news_data.csv
> - Provided news dataset for clustering
>
